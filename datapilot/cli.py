"""
Typer CLI entry.

Subcommands:
  datapilot               new or resumed session (default subcommand)
  datapilot list          list saved projects
  datapilot config show   print effective config (no secrets)
  datapilot config path   print the config file path
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from datapilot.agent.setup import run_wizard
from datapilot.config import (
    DEFAULT_ROOT,
    Config,
    config_path,
    load_config,
    resolve_agent_override,
    save_config,
)
from datapilot.security.log_filter import install as install_log_filter
from datapilot.tui.shell import TUIShell

app = typer.Typer(add_completion=False, no_args_is_help=False, help="DataPilot — terminal AI copilot for data science / analysis.")
config_app = typer.Typer(help="Inspect or reset configuration.")
app.add_typer(config_app, name="config")


def _bootstrap() -> Config:
    install_log_filter()
    cfg = load_config()
    if not cfg.first_run_completed:
        console = Console()
        cfg = run_wizard(cfg, console=console)
    cfg.agent = resolve_agent_override(cfg.agent)
    return cfg


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context):
    if ctx.invoked_subcommand is not None:
        return
    cfg = _bootstrap()
    TUIShell(cfg).run()


@app.command("list")
def list_projects():
    """List saved projects under the storage root."""
    cfg = load_config()
    console = Console()
    projects_dir = cfg.projects_dir
    if not projects_dir.exists():
        console.print("[dim]No projects yet.[/dim]")
        return
    table = Table(title=f"Projects in {projects_dir}")
    table.add_column("Name")
    table.add_column("Updated")
    for p in sorted(projects_dir.iterdir()):
        if p.is_dir():
            session = p / "session.json"
            mtime = session.stat().st_mtime if session.exists() else p.stat().st_mtime
            from datetime import datetime
            table.add_row(p.name, datetime.fromtimestamp(mtime).isoformat(timespec="seconds"))
    console.print(table)


@config_app.command("show")
def config_show():
    cfg = load_config()
    console = Console()
    console.print(f"[bold]agent[/bold]      {cfg.agent.provider} / {cfg.agent.model}")
    console.print(f"[bold]temperature[/bold] {cfg.agent.temperature}")
    console.print(f"[bold]privacy[/bold]    cloud_mode={cfg.data_privacy.cloud_mode} "
                  f"pii_masking={cfg.data_privacy.pii_masking}")
    console.print(f"[bold]storage[/bold]    {cfg.storage_root}")
    console.print(f"[bold]first run[/bold]  completed={cfg.first_run_completed}")


@config_app.command("path")
def config_path_cmd():
    print(config_path(DEFAULT_ROOT))


if __name__ == "__main__":
    app()
