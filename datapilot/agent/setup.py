"""
First-run agent-selection wizard.

Runs exactly once on first launch (or on demand via the /agent slash command).
Persists the user's choice in ~/.datapilot/config.json and stores any API key
in the OS keyring — never in the config file.
"""

from __future__ import annotations

from getpass import getpass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from datapilot.config import (
    AgentConfig,
    CLOUD_PROVIDERS,
    Config,
    PROVIDER_DEFAULTS,
    save_config,
)
from datapilot.security import keyring_store

_PROVIDER_ORDER = ["ollama", "anthropic", "openai", "google"]


def run_wizard(cfg: Config, *, console: Console | None = None) -> Config:
    """Interactive wizard. Mutates + returns the passed Config, persisted."""
    console = console or Console()

    _render_menu(console)
    choice = _ask_provider(console)
    provider = _PROVIDER_ORDER[choice - 1]
    default_model = PROVIDER_DEFAULTS[provider]["model"]

    model = _ask_model(console, default_model)

    if provider in CLOUD_PROVIDERS:
        _ensure_api_key(console, provider)

    cfg.agent = AgentConfig(
        provider=provider,
        model=model,
        temperature=cfg.agent.temperature,
        max_tokens=cfg.agent.max_tokens,
    )
    cfg.first_run_completed = True
    save_config(cfg)

    console.print(
        f"\n[green]Saved.[/green] Default agent set to "
        f"[bold]{provider}[/bold] / [bold]{model}[/bold]. "
        "Change any time with [cyan]/agent[/cyan].\n"
    )
    return cfg


def _render_menu(console: Console) -> None:
    table = Table(title="Pick your default AI agent", show_header=False, box=None)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Provider")
    for i, key in enumerate(_PROVIDER_ORDER, start=1):
        table.add_row(str(i), PROVIDER_DEFAULTS[key]["label"])
    console.print(
        Panel.fit(
            table,
            title="DataPilot",
            subtitle="one-time setup — change any time with /agent",
        )
    )


def _ask_provider(console: Console) -> int:
    while True:
        raw = console.input("[bold]>[/bold] Pick 1-4: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(_PROVIDER_ORDER):
            return int(raw)
        console.print("[red]Please enter 1, 2, 3, or 4.[/red]")


def _ask_model(console: Console, default: str) -> str:
    raw = console.input(f"[bold]>[/bold] Model [dim]({default})[/dim]: ").strip()
    return raw or default


def _ensure_api_key(console: Console, provider: str) -> None:
    """Import from env if available; otherwise prompt (hidden input)."""
    if keyring_store.import_env_key(provider):
        var = keyring_store.env_var_for(provider)
        console.print(
            f"[green]Imported {var} into the OS keyring.[/green] "
            "The key is no longer read from your environment."
        )
        return

    if keyring_store.get_key(provider):
        console.print(
            f"[green]Existing {provider} key found in keyring — kept.[/green] "
            "Use [cyan]/agent[/cyan] to rotate it."
        )
        return

    console.print(
        f"[yellow]Enter your {provider} API key.[/yellow] "
        "It will be stored in your OS keyring — never in a file."
    )
    while True:
        key = getpass("    API key (input hidden): ").strip()
        if key:
            break
        console.print("[red]Key must not be empty.[/red]")
    keyring_store.set_key(provider, key)
    console.print("[green]Key saved.[/green]")
