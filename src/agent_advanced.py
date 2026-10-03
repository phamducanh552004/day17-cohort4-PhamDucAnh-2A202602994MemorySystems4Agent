from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Short-term memory, persistent User.md, and compacted long context."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(self.config.compact_threshold_tokens, self.config.compact_keep_messages)
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "user", message)
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + estimate_tokens(answer)
        return {"answer": answer, "token_usage": self.thread_tokens[thread_id], "prompt_tokens_processed": self.thread_prompt_tokens[thread_id], "memory_path": str(self.profile_store.path_for(user_id))}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        messages = context["messages"]
        assert isinstance(messages, list)
        combined = self.profile_store.read_text(user_id) + str(context["summary"]) + " ".join(str(item["content"]) for item in messages)
        return estimate_tokens(combined)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        lowered = message.lower()
        if any(word in lowered for word in ("tên", "ở đâu", "nghề", "style", "đồ uống", "món ăn", "nuôi", "nhắc lại", "tóm tắt", "ai đó nhắc")):
            if not facts:
                return "Mình chưa có hồ sơ dài hạn của bạn."
            labels = {"name": "Tên", "location": "Nơi ở", "profession": "Nghề nghiệp", "response_style": "Style", "favorite_drink": "Đồ uống", "favorite_food": "Món ăn", "pet": "Thú cưng", "interests": "Mối quan tâm"}
            return "; ".join(f"{labels.get(key, key)}: {value}" for key, value in facts.items())
        return "Mình đã cập nhật hồ sơ dài hạn và giữ ngữ cảnh gần đây."

    def _maybe_build_langchain_agent(self):
        return None
