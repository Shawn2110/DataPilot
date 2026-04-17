"""
LLM provider factory.

Each provider is constructed from an AgentConfig + an API key loaded from the
keyring (for cloud providers). Ollama runs locally and needs no key.
"""

from __future__ import annotations

from datapilot.config import AgentConfig, CLOUD_PROVIDERS
from datapilot.security import keyring_store


class MissingAPIKeyError(RuntimeError):
    pass


class UnsupportedProviderError(RuntimeError):
    pass


def create_llm(cfg: AgentConfig):
    """
    Return a LangChain chat model for the configured provider.

    Imports are lazy: the user only needs the SDK for the provider they picked.
    """
    provider = cfg.provider.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=cfg.model, temperature=cfg.temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        key = _require_key(provider)
        return ChatAnthropic(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            api_key=key,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        key = _require_key(provider)
        return ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            api_key=key,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = _require_key(provider)
        return ChatGoogleGenerativeAI(
            model=cfg.model,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens,
            google_api_key=key,
        )

    raise UnsupportedProviderError(f"Unknown provider: {cfg.provider!r}")


def _require_key(provider: str) -> str:
    if provider not in CLOUD_PROVIDERS:
        return ""
    key = keyring_store.get_key(provider)
    if not key:
        raise MissingAPIKeyError(
            f"No API key stored for {provider!r}. Run /agent to (re-)configure."
        )
    return key
