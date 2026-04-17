"""
Session + Cell dataclasses.

A session is persisted as session.json after every cell. A cell is the
smallest unit of work — a markdown narration + a code block, plus the
DataFrame snapshot taken BEFORE the cell ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

CELL_STATUS = ("pending", "ran", "failed", "edited", "stale")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Cell:
    id: int
    narration: str = ""
    code: str = ""
    output: str = ""
    error: str | None = None
    status: str = "pending"
    snapshot_path: Path | None = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def touch(self) -> None:
        self.updated_at = now_iso()


@dataclass
class Session:
    id: str
    name: str
    mode: str                           # "data-analysis" | "data-science"
    project_dir: Path
    agent_provider: str
    agent_model: str
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    cells: list[Cell] = field(default_factory=list)
    current_csv_path: str | None = None

    def next_cell_id(self) -> int:
        return (max((c.id for c in self.cells), default=0)) + 1

    def find(self, cell_id: int) -> Cell | None:
        for c in self.cells:
            if c.id == cell_id:
                return c
        return None

    def touch(self) -> None:
        self.updated_at = now_iso()
