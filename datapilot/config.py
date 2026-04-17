"""
User configuration, stored at ~/.datapilot/config.json.

Contains no secrets. API keys live in the OS keyring (see
datapilot.security.keyring_store). The config file is created on first
launch with mode 0600.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_VERSION = 1

DEFAULT_ROOT = Path.home() / ".datapilot"
CONFIG_FILENAME = "config.json"

# Supported provider IDs. Models are suggested; the user can override.
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "ollama": {"model": "llama3.1", "label": "Ollama / Llama 3.1 (free, local, offline)"},
    "anthropic": {"model": "claude-opus-4-7", "label": "Anthropic / Claude"},
    "openai": {"model": "gpt-4o", "label": "OpenAI / GPT-4o"},
    "google": {"model": "gemini-2.5-pro", "label": "Google / Gemini"},
}

# Providers whose calls leave the user's machine.
CLOUD_PROVIDERS = frozenset({"anthropic", "openai", "google"})


@dataclass
class AgentConfig:
    provider: str = "ollama"
    model: str = "llama3.1"
    temperature: float = 0.2
    max_tokens: int = 4096


@dataclass
class PrivacyConfig:
    # "schema_only" transmits column names + dtypes + summary stats to cloud LLMs.
    # "full_data" additionally allows sample rows (still with PII masking).
    cloud_mode: str = "schema_only"
    pii_masking: bool = True


@dataclass
class Config:
    version: int = CONFIG_VERSION
    first_run_completed: bool = False
    agent: AgentConfig = field(default_factory=AgentConfig)
    data_privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    storage_root: str = str(DEFAULT_ROOT)

    @property
    def projects_dir(self) -> Path:
        return Path(self.storage_root).expanduser() / "projects"


def _ensure_root(root: Path) -> None:
    """Create ~/.datapilot/ with mode 0700 if missing."""
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except (OSError, NotImplementedError):
        pass


def _atomic_write(path: Path, data: str, mode: int = 0o600) -> None:
    """Write-rename with fsync; prevents torn writes on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        try:
            os.chmod(path, mode)
        except (OSError, NotImplementedError):
            pass
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def config_path(root: Path | None = None) -> Path:
    return (root or DEFAULT_ROOT) / CONFIG_FILENAME


def load_config(root: Path | None = None) -> Config:
    root = root or DEFAULT_ROOT
    _ensure_root(root)
    path = config_path(root)
    if not path.exists():
        return Config(storage_root=str(root))

    raw = json.loads(path.read_text(encoding="utf-8"))
    agent_raw = raw.get("agent", {})
    privacy_raw = raw.get("data_privacy", {})
    return Config(
        version=raw.get("version", CONFIG_VERSION),
        first_run_completed=bool(raw.get("first_run_completed", False)),
        agent=AgentConfig(
            provider=agent_raw.get("provider", "ollama"),
            model=agent_raw.get("model", "llama3.1"),
            temperature=float(agent_raw.get("temperature", 0.2)),
            max_tokens=int(agent_raw.get("max_tokens", 4096)),
        ),
        data_privacy=PrivacyConfig(
            cloud_mode=privacy_raw.get("cloud_mode", "schema_only"),
            pii_masking=bool(privacy_raw.get("pii_masking", True)),
        ),
        storage_root=raw.get("storage_root", str(root)),
    )


def save_config(cfg: Config, root: Path | None = None) -> None:
    root = Path(cfg.storage_root).expanduser() if cfg.storage_root else (root or DEFAULT_ROOT)
    _ensure_root(root)
    data = json.dumps(asdict(cfg), indent=2)
    _atomic_write(config_path(root), data, mode=0o600)


def resolve_agent_override(cfg: AgentConfig) -> AgentConfig:
    """
    Apply the DATAPILOT_AGENT env var override, if any.
    Format: 'provider' or 'provider:model'. Does not persist.
    """
    raw = os.environ.get("DATAPILOT_AGENT", "").strip()
    if not raw:
        return cfg
    provider, _, model = raw.partition(":")
    provider = provider.lower()
    if provider not in PROVIDER_DEFAULTS:
        return cfg
    return AgentConfig(
        provider=provider,
        model=model or PROVIDER_DEFAULTS[provider]["model"],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
