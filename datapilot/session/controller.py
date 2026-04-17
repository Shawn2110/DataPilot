"""
SessionController: orchestrates the REPL loop.

Responsibilities:
  - Proxy user input to the TUI.
  - Route slash commands.
  - Propose a step via the agent, show it, wait for user approval.
  - On approval, validate + execute the code in the restricted namespace.
  - After every cell: atomic-save session.json, regenerate notebook.ipynb,
    write a parquet snapshot taken BEFORE the cell ran.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from datapilot.execution.executor import ExecResult, execute
from datapilot.session.models import Cell, Session
from datapilot.storage.manager import StorageManager


class SessionController:
    def __init__(self, session: Session, storage: StorageManager):
        self.session = session
        self.storage = storage
        self._namespace: dict[str, Any] = {}
        if session.current_csv_path:
            self._try_load_initial_df(Path(session.current_csv_path))

    # ----- public API -----------------------------------------------------
    def add_proposed_cell(self, narration: str, code: str) -> Cell:
        cell = Cell(
            id=self.session.next_cell_id(),
            narration=narration,
            code=code,
            status="pending",
        )
        self.session.cells.append(cell)
        self.session.touch()
        self._persist()
        return cell

    def run_cell(self, cell: Cell, project_dir: Path) -> ExecResult:
        if not cell.code.strip():
            cell.status = "ran"
            cell.touch()
            self.session.touch()
            self._persist()
            return ExecResult(ok=True)

        df = self._namespace.get("df")
        if isinstance(df, pd.DataFrame):
            snap_path = self.storage.save_snapshot(cell.id, df)
            cell.snapshot_path = snap_path

        result = execute(cell.code, project_dir=project_dir, namespace=self._namespace)
        cell.output = (result.stdout or "")[:10_000]
        if result.ok:
            cell.status = "ran"
            cell.error = None
        else:
            cell.status = "failed"
            cell.error = result.error or "validation failed"
        cell.touch()
        self.session.touch()
        self._persist()
        return result

    def rerun_from(self, cell_id: int) -> None:
        target = self.session.find(cell_id)
        if not target:
            raise ValueError(f"No such cell: {cell_id}")
        snap = self.storage.load_snapshot(cell_id)
        if snap is not None:
            self._namespace["df"] = snap
        start = next(i for i, c in enumerate(self.session.cells) if c.id == cell_id)
        for cell in self.session.cells[start:]:
            self.run_cell(cell, self.session.project_dir)

    def revert_from(self, cell_id: int) -> int:
        keep = [c for c in self.session.cells if c.id < cell_id]
        dropped = len(self.session.cells) - len(keep)
        self.session.cells = keep
        self.session.touch()
        self._persist()
        return dropped

    # ----- internals ------------------------------------------------------
    def _persist(self) -> None:
        self.storage.save_session(self.session)
        try:
            self.storage.save_notebook(self.session)
        except Exception:
            # Notebook export is best-effort; session.json is the source of truth.
            pass

    def _try_load_initial_df(self, csv_path: Path) -> None:
        try:
            if csv_path.suffix.lower() in {".xlsx", ".xls"}:
                self._namespace["df"] = pd.read_excel(csv_path)
            else:
                self._namespace["df"] = pd.read_csv(csv_path)
        except Exception:
            pass
