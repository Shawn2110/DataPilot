"""
Slash-command registry.

Each command is a small handler taking (shell, arg_str) -> None. Keeping
handlers tiny keeps the router simple and testable.
"""

from __future__ import annotations

from typing import Callable, Dict


CommandHandler = Callable[["TUIShell", str], None]  # noqa: F821 - forward ref


class CommandRegistry:
    def __init__(self):
        self._commands: Dict[str, tuple[CommandHandler, str]] = {}

    def register(self, name: str, handler: CommandHandler, help_text: str) -> None:
        self._commands[name] = (handler, help_text)

    def run(self, shell, raw: str) -> bool:
        """Returns True if the input was a recognised command."""
        if not raw.startswith("/"):
            return False
        name, _, rest = raw[1:].partition(" ")
        name = name.strip().lower()
        entry = self._commands.get(name)
        if not entry:
            shell.console.print(f"[red]Unknown command:[/red] /{name}. Try /help.")
            return True
        handler, _ = entry
        handler(shell, rest.strip())
        return True

    def help_text(self) -> list[tuple[str, str]]:
        return [(f"/{name}", desc) for name, (_, desc) in sorted(self._commands.items())]
