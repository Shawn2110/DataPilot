"""
API-key storage.

Primary: OS keyring (macOS Keychain, Windows Credential Manager, Linux Secret
Service). Fallback: Fernet-encrypted file under ~/.datapilot/ keyed by a
machine-local secret. Keys are never written to config.json or the project dir.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from pathlib import Path

from datapilot.config import DEFAULT_ROOT, _atomic_write

SERVICE_NAME = "datapilot"
_FALLBACK_VAULT = "fallback_keys.enc"
_FALLBACK_KEY = "fallback.key"


def _fallback_paths(root: Path) -> tuple[Path, Path]:
    return root / _FALLBACK_KEY, root / _FALLBACK_VAULT


def _get_or_create_fallback_key(key_path: Path) -> bytes:
    if key_path.exists():
        return key_path.read_bytes()
    raw = secrets.token_bytes(32)
    key = base64.urlsafe_b64encode(raw)
    _atomic_write(key_path, key.decode("ascii"), mode=0o600)
    return key


def _load_fallback_vault(root: Path) -> dict[str, str]:
    from cryptography.fernet import Fernet, InvalidToken

    key_path, vault_path = _fallback_paths(root)
    if not vault_path.exists():
        return {}
    key = _get_or_create_fallback_key(key_path)
    try:
        decrypted = Fernet(key).decrypt(vault_path.read_bytes())
    except InvalidToken:
        return {}
    return json.loads(decrypted.decode("utf-8"))


def _save_fallback_vault(root: Path, data: dict[str, str]) -> None:
    from cryptography.fernet import Fernet

    key_path, vault_path = _fallback_paths(root)
    key = _get_or_create_fallback_key(key_path)
    encrypted = Fernet(key).encrypt(json.dumps(data).encode("utf-8"))
    _atomic_write(vault_path, encrypted.decode("latin-1"), mode=0o600)


def set_key(provider: str, api_key: str, root: Path | None = None) -> None:
    """Store the API key for a provider. Never logs the value."""
    root = root or DEFAULT_ROOT
    try:
        import keyring

        keyring.set_password(SERVICE_NAME, provider, api_key)
        return
    except Exception:
        pass

    vault = _load_fallback_vault(root)
    vault[provider] = api_key
    _save_fallback_vault(root, vault)


def get_key(provider: str, root: Path | None = None) -> str | None:
    root = root or DEFAULT_ROOT
    try:
        import keyring

        value = keyring.get_password(SERVICE_NAME, provider)
        if value:
            return value
    except Exception:
        pass

    vault = _load_fallback_vault(root)
    return vault.get(provider)


def delete_key(provider: str, root: Path | None = None) -> None:
    root = root or DEFAULT_ROOT
    try:
        import keyring

        keyring.delete_password(SERVICE_NAME, provider)
    except Exception:
        pass

    vault = _load_fallback_vault(root)
    if provider in vault:
        del vault[provider]
        _save_fallback_vault(root, vault)


def env_var_for(provider: str) -> str | None:
    """Standard environment variable for a provider's key, if any."""
    return {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(provider)


def import_env_key(provider: str, root: Path | None = None) -> bool:
    """
    If the provider's env var is set, import it into the keyring and return True.
    Used by the first-run wizard to avoid re-prompting when the user already has
    a key in their environment.
    """
    var = env_var_for(provider)
    if not var:
        return False
    value = os.environ.get(var)
    if not value:
        return False
    set_key(provider, value, root=root)
    return True
