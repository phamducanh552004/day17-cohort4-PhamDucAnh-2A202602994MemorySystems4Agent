from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Within-thread memory only; fresh threads deliberately forget facts."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        prompt_tokens = estimate_tokens(" ".join(item["content"] for item in session.messages) + " " + message)
        session.prompt_tokens_processed += prompt_tokens
        if len(session.messages) == 0 and any(word in message.lower() for word in ("tên", "ở đâu", "nghề", "style", "đồ uống")):
            answer = "Mình chưa có thông tin từ thread này để trả lời."
        else:
            answer = "Mình đã ghi nhận trong cuộc hội thoại hiện tại."
        session.messages.extend([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
        session.token_usage += estimate_tokens(answer)
        return {"answer": answer, "token_usage": session.token_usage, "prompt_tokens_processed": session.prompt_tokens_processed}

    def _maybe_build_langchain_agent(self):
        return None
