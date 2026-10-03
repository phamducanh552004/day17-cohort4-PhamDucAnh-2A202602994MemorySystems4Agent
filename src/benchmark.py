from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    return sum(item.casefold() in answer.casefold() for item in expected) / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    return recall_points(answer, expected)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    tokens = prompts = compactions = 0
    scores: list[float] = []
    memory_growth = 0
    for conversation in conversations:
        user_id, thread_id = conversation["user_id"], conversation["id"]
        for turn in conversation["turns"]:
            agent.reply(user_id, thread_id, turn)
        tokens += agent.token_usage(thread_id)
        prompts += agent.prompt_token_usage(thread_id)
        compactions += agent.compaction_count(thread_id)
        if hasattr(agent, "memory_file_size"):
            memory_growth += agent.memory_file_size(user_id)
        recall_thread = f"{thread_id}-recall"
        for recall in conversation.get("recall_questions", []):
            answer = agent.reply(user_id, recall_thread, recall["question"])["answer"]
            score = recall_points(answer, recall["expected_contains"])
            scores.append(score)
            tokens += agent.token_usage(recall_thread)
            prompts += agent.prompt_token_usage(recall_thread)
    average = sum(scores) / len(scores) if scores else 0.0
    return BenchmarkRow(agent_name, tokens, prompts, average, average, memory_growth, compactions)


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = ["Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall", "Response quality", "Memory growth (bytes)", "Compactions"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = [row.agent_name, str(row.agent_tokens_only), str(row.prompt_tokens_processed), f"{row.recall_score:.3f}", f"{row.response_quality:.3f}", str(row.memory_growth_bytes), str(row.compactions)]
        lines.append("| " + " | ".join(values) + " |")
    return chr(10).join(lines)


def _agents_for(config, label: str):
    baseline_config = replace(config, state_dir=config.state_dir / f"{label}_baseline")
    advanced_config = replace(config, state_dir=config.state_dir / f"{label}_advanced")
    return BaselineAgent(baseline_config, force_offline=True), AdvancedAgent(advanced_config, force_offline=True)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    for title, filename, label in (
        ("Standard Benchmark", "conversations.json", "standard"),
        ("Long-Context Stress Benchmark", "advanced_long_context.json", "stress"),
    ):
        conversations = load_conversations(config.data_dir / filename)
        baseline, advanced = _agents_for(config, label)
        rows = [
            run_agent_benchmark("Baseline", baseline, conversations, config),
            run_agent_benchmark("Advanced", advanced, conversations, config),
        ]
        print(title)
        print(format_rows(rows))
        print()


if __name__ == "__main__":
    main()
