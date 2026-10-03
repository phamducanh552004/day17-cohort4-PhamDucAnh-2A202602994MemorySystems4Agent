from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def make_config(tmp_path: Path):
    config = load_config(tmp_path)
    return replace(config, compact_threshold_tokens=30, compact_keep_messages=2)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.profile_store.write_text("u1", "# User Profile" + chr(10) + "- name: An")
    assert agent.profile_store.edit_text("u1", "An", "Anh")
    assert "Anh" in agent.profile_store.read_text("u1")


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for _ in range(5):
        agent.reply("u1", "thread", "Mình thích Python và đây là một tin nhắn đủ dài để kích hoạt compact memory.")
    assert agent.compaction_count("thread") > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    fact = "Mình tên là DũngCT và hiện đang làm MLOps engineer."
    advanced.reply("u1", "old", fact)
    baseline.reply("u1", "old", fact)
    assert "DũngCT" in advanced.reply("u1", "new", "Mình tên gì?")["answer"]
    assert "DũngCT" not in baseline.reply("u1", "new", "Mình tên gì?")["answer"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for _ in range(8):
        text = "Mình đang gửi một đoạn hội thoại dài về Python, MLOps và memory compaction để so sánh prompt load."
        baseline.reply("u1", "long", text)
        advanced.reply("u1", "long", text)
    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
