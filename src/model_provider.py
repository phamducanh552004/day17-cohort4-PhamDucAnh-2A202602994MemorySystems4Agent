from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    aliases = {"anthorpic": "anthropic", "google": "gemini", "open-router": "openrouter", "openai-compatible": "custom"}
    return aliases.get(value.strip().lower(), value.strip().lower())


def build_chat_model(config: ProviderConfig):
    provider = normalize_provider(config.provider)
    if provider in {"openai", "custom", "openrouter"}:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.model_name, api_key=config.api_key, base_url=config.base_url or ("https://openrouter.ai/api/v1" if provider == "openrouter" else None), temperature=config.temperature)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=config.model_name, google_api_key=config.api_key, temperature=config.temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=config.model_name, api_key=config.api_key, temperature=config.temperature)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=config.model_name, base_url=config.base_url, temperature=config.temperature)
    raise ValueError(f"Unsupported provider: {config.provider}")
