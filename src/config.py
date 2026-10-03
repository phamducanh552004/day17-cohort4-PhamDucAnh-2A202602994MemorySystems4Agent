from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _provider_from_env(prefix: str = "LLM") -> ProviderConfig:
    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", "openai"))
    model = os.getenv(f"{prefix}_MODEL", "gpt-4o-mini")
    key_name = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY", "custom": "CUSTOM_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
    return ProviderConfig(provider, model, api_key=os.getenv(key_name) if key_name else None, base_url=os.getenv(f"{provider.upper()}_BASE_URL") or os.getenv("CUSTOM_BASE_URL"))


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env")
    except ImportError:
        pass
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(root, root / "data", state_dir, int(os.getenv("COMPACT_THRESHOLD_TOKENS", "300")), int(os.getenv("COMPACT_KEEP_MESSAGES", "6")), _provider_from_env(), _provider_from_env("JUDGE"))
