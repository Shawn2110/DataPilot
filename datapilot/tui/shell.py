"""
TUI shell: the top-level REPL.

Minimal-but-usable first pass. Uses Rich for rendering and prompt_toolkit for
line editing + history. The shell holds the SessionController and the
CommandRegistry; it does not itself make LLM calls — that lives in runner.py.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from datapilot.agent.setup import run_wizard
from datapilot.config import Config, load_config, save_config
from datapilot.session.controller import SessionController
from datapilot.session.models import Session
from datapilot.storage.manager import StorageManager
from datapilot.storage.paths import ensure_project_dir, slugify
from datapilot.tui.commands import CommandRegistry


class TUIShell:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.console = Console()
        self.prompt = PromptSession(history=InMemoryHistory())
        self.commands = CommandRegistry()
        self.controller: SessionController | None = None
        self._register_builtin_commands()

    # ----- public API -----------------------------------------------------
    def run(self) -> None:
        self._welcome()
        self._bootstrap_session()
        while True:
            try:
                raw = self.prompt.prompt(" > ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]bye[/dim]")
                return
            if not raw:
                continue
            if not self.commands.run(self, raw):
                self.console.print(
                    "[dim]Type /help for commands. "
                    "Free-text prompting of the agent is coming in the next build step.[/dim]"
                )

    # ----- welcome + session bootstrap -----------------------------------
    def _welcome(self) -> None:
        self.console.print(
            Panel.fit(
                "[bold]DataPilot[/bold]\nAI Data Science & Analysis Copilot",
                border_style="cyan",
            )
        )
        self.console.print(
            f"[dim]agent:[/dim] {self.cfg.agent.provider} / {self.cfg.agent.model}   "
            f"[dim]privacy:[/dim] {self.cfg.data_privacy.cloud_mode}   "
            f"[dim]storage:[/dim] {self.cfg.storage_root}"
        )

    def _bootstrap_session(self) -> None:
        name = self.console.input("Project name: ").strip() or "untitled"
        slug = slugify(name)
        project_dir = ensure_project_dir(self.cfg.projects_dir, slug)

        mode = self._ask_mode()
        csv = self.console.input("Path to CSV (blank to skip): ").strip() or None

        session = Session(
            id=str(uuid.uuid4()),
            name=name,
            mode=mode,
            project_dir=project_dir,
            agent_provider=self.cfg.agent.provider,
            agent_model=self.cfg.agent.model,
            current_csv_path=csv,
        )
        storage = StorageManager(project_dir)
        self.controller = SessionController(session, storage)
        storage.save_session(session)
        self.console.print(
            f"[green]Started session[/green] [bold]{slug}[/bold] "
            f"in mode [cyan]{mode}[/cyan]. Project at {project_dir}."
        )

    def _ask_mode(self) -> str:
        while True:
            raw = self.console.input("Mode: [1] Data Analysis  [2] Data Science > ").strip()
            if raw in {"1", "a", "analysis", "data-analysis"}:
                return "data-analysis"
            if raw in {"2", "s", "science", "data-science"}:
                return "data-science"
            self.console.print("[red]Pick 1 or 2.[/red]")

    # ----- command handlers ----------------------------------------------
    def _register_builtin_commands(self) -> None:
        self.commands.register("help", _cmd_help, "List all commands")
        self.commands.register("quit", _cmd_quit, "Exit the session")
        self.commands.register("cells", _cmd_cells, "List cells in this session")
        self.commands.register("agent", _cmd_agent, "Re-run the agent-selection wizard")
        self.commands.register("mode", _cmd_mode, "Show or switch the current mode")
        self.commands.register("code", _cmd_code, "Show the code for a given cell")
        self.commands.register("save", _cmd_save, "Force-save the current session")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
def _cmd_help(shell: TUIShell, _arg: str) -> None:
    table = Table(title="Commands", show_header=True, header_style="bold")
    table.add_column("Command")
    table.add_column("Description")
    for name, desc in shell.commands.help_text():
        table.add_row(name, desc)
    shell.console.print(table)


def _cmd_quit(shell: TUIShell, _arg: str) -> None:
    raise EOFError()


def _cmd_cells(shell: TUIShell, _arg: str) -> None:
    if not shell.controller or not shell.controller.session.cells:
        shell.console.print("[dim]No cells yet.[/dim]")
        return
    table = Table(show_header=True)
    table.add_column("#", justify="right")
    table.add_column("Status")
    table.add_column("Narration (preview)")
    for cell in shell.controller.session.cells:
        preview = (cell.narration or "").splitlines()[0][:60] if cell.narration else ""
        table.add_row(str(cell.id), cell.status, preview)
    shell.console.print(table)


def _cmd_agent(shell: TUIShell, _arg: str) -> None:
    cfg = run_wizard(shell.cfg, console=shell.console)
    shell.cfg = cfg
    if shell.controller:
        shell.controller.session.agent_provider = cfg.agent.provider
        shell.controller.session.agent_model = cfg.agent.model
        shell.controller.storage.save_session(shell.controller.session)


def _cmd_mode(shell: TUIShell, arg: str) -> None:
    if not shell.controller:
        shell.console.print("[yellow]No active session.[/yellow]")
        return
    if not arg:
        shell.console.print(f"Current mode: [cyan]{shell.controller.session.mode}[/cyan]")
        return
    arg_l = arg.strip().lower()
    if arg_l in {"1", "a", "analysis", "data-analysis"}:
        shell.controller.session.mode = "data-analysis"
    elif arg_l in {"2", "s", "science", "data-science"}:
        shell.controller.session.mode = "data-science"
    else:
        shell.console.print("[red]Unknown mode. Use data-analysis or data-science.[/red]")
        return
    shell.controller.storage.save_session(shell.controller.session)
    shell.console.print(f"[green]Mode set to[/green] {shell.controller.session.mode}")


def _cmd_code(shell: TUIShell, arg: str) -> None:
    if not shell.controller:
        shell.console.print("[yellow]No active session.[/yellow]")
        return
    if not arg.strip().isdigit():
        shell.console.print("Usage: /code <cell-id>")
        return
    cell = shell.controller.session.find(int(arg))
    if not cell:
        shell.console.print(f"[red]No such cell:[/red] {arg}")
        return
    if not cell.code.strip():
        shell.console.print("[dim](this cell has no code)[/dim]")
        return
    shell.console.print(Syntax(cell.code, "python", theme="monokai", line_numbers=True))


def _cmd_save(shell: TUIShell, _arg: str) -> None:
    if not shell.controller:
        shell.console.print("[yellow]No active session.[/yellow]")
        return
    shell.controller.storage.save_session(shell.controller.session)
    shell.controller.storage.save_notebook(shell.controller.session)
    shell.console.print("[green]Saved.[/green]")
