from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    return (len(text.strip()) + 3) // 4 if text.strip() else 0


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "anonymous"


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / _safe_id(user_id) / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User Profile" + chr(10)

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + chr(10), encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return dict(re.findall(r"^- ([^:]+): (.+)$", self.read_text(user_id), re.MULTILINE))

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        facts = self.facts(user_id)
        facts[key] = value
        lines = ["# User Profile", *(f"- {fact}: {detail}" for fact, detail in facts.items())]
        self.write_text(user_id, chr(10).join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    text = message.strip()
    if "?" in text:
        return {}
    updates: dict[str, str] = {}
    patterns = {
        "name": r"(?:mình tên là|tên mình là)\s+([^,.!?]+)",
        "location": r"(?:hiện (?:tại )?mình (?:đang )?(?:ở|làm việc ở)|mình (?:đang )?ở|đang làm việc ở)\s+([^,.!?]+)",
        "profession": r"(?:giờ (?:chuyển sang|là)|nghề nghiệp hiện tại (?:vẫn là|là)|đang làm)\s+([^,.!?]+?(?:engineer|manager|developer|architect|designer))",
        "favorite_drink": r"(?:đồ uống yêu thích là|vẫn uống)\s+([^,.!?]+)",
        "favorite_food": r"(?:món ăn yêu thích là|buổi trưa mình ăn)\s+([^,.!?]+)",
        "pet": r"(?:nuôi (?:một |mình )?(?:bé |con )?)([^,.!?]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            updates[key] = match.group(1).strip()
    lower = text.lower()
    if "3 bullet" in lower:
        updates["response_style"] = "3 bullet ngắn, có ví dụ thực chiến"
    elif "trả lời ngắn gọn" in lower or "bullet ngắn" in lower:
        updates["response_style"] = "ngắn gọn, có bullet và ví dụ thực tế"
    if "python" in lower and ("thích" in lower or "quan tâm" in lower):
        updates["interests"] = "Python và AI ứng dụng"
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    return " | ".join(f"{item['role']}: {item['content']}" for item in messages[-max_items:])


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        all_text = str(thread["summary"]) + " " + summarize_messages(messages, len(messages))
        if estimate_tokens(all_text) > self.threshold_tokens and len(messages) > self.keep_messages:
            older, recent = messages[:-self.keep_messages], messages[-self.keep_messages:]
            thread["summary"] = summarize_messages(older, max_items=4)
            thread["messages"] = recent
            thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self._thread(thread_id)
        return {"messages": list(thread["messages"]), "summary": str(thread["summary"]), "compactions": int(thread["compactions"])}

    def compaction_count(self, thread_id: str) -> int:
        return int(self._thread(thread_id)["compactions"])
