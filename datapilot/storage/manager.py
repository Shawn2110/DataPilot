"""
StorageManager: atomic session saves + DataFrame snapshots + notebook export.

After every generated cell the controller calls save_session() — which
regenerates session.json, notebook.ipynb, and (if the cell ran code) a parquet
snapshot. All writes are atomic (tempfile + fsync + rename).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from datapilot.config import _atomic_write
from datapilot.session.models import Cell, Session
from datapilot.storage import paths as P

try:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
except Exception:  # pragma: no cover - imported lazily in save_notebook
    nbformat = None  # type: ignore


class StorageManager:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir

    # ----- session.json --------------------------------------------------
    def save_session(self, session: Session) -> None:
        data = json.dumps(_session_to_json(session), indent=2)
        _atomic_write(P.session_path(self.project_dir), data, mode=0o600)

    def load_session(self) -> Session | None:
        path = P.session_path(self.project_dir)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _session_from_json(raw, self.project_dir)

    # ----- notebook.ipynb ------------------------------------------------
    def save_notebook(self, session: Session) -> None:
        if nbformat is None:
            raise RuntimeError(
                "nbformat is required for notebook export. "
                "Run `pip install nbformat`."
            )
        nb = new_notebook()
        for cell in session.cells:
            if cell.narration:
                nb.cells.append(new_markdown_cell(cell.narration))
            if cell.code:
                nb.cells.append(new_code_cell(cell.code))
        path = P.notebook_path(self.project_dir)
        nbformat.write(nb, str(path))
        try:
            path.chmod(0o600)
        except OSError:
            pass

    # ----- DataFrame snapshots -------------------------------------------
    def save_snapshot(self, cell_id: int, df: pd.DataFrame) -> Path:
        """Snapshot the DataFrame state BEFORE cell_id runs. Zstd-compressed."""
        path = P.snapshot_path(self.project_dir, cell_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path, compression="zstd", index=False)
        except Exception:
            df.to_parquet(path, index=False)
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return path

    def load_snapshot(self, cell_id: int) -> pd.DataFrame | None:
        path = P.snapshot_path(self.project_dir, cell_id)
        if not path.exists():
            return None
        return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _session_to_json(session: Session) -> dict:
    d = asdict(session)
    d["project_dir"] = str(session.project_dir)
    d["cells"] = [
        {
            **asdict(cell),
            "snapshot_path": str(cell.snapshot_path) if cell.snapshot_path else None,
        }
        for cell in session.cells
    ]
    return d


def _session_from_json(raw: dict, project_dir: Path) -> Session:
    cells = [
        Cell(
            id=int(c["id"]),
            narration=c.get("narration", ""),
            code=c.get("code", ""),
            output=c.get("output", ""),
            error=c.get("error"),
            status=c.get("status", "pending"),
            snapshot_path=Path(c["snapshot_path"]) if c.get("snapshot_path") else None,
            created_at=c.get("created_at", ""),
            updated_at=c.get("updated_at", ""),
        )
        for c in raw.get("cells", [])
    ]
    return Session(
        id=raw["id"],
        name=raw["name"],
        mode=raw["mode"],
        project_dir=project_dir,
        agent_provider=raw.get("agent_provider", "ollama"),
        agent_model=raw.get("agent_model", "llama3.1"),
        created_at=raw.get("created_at", ""),
        updated_at=raw.get("updated_at", ""),
        cells=cells,
        current_csv_path=raw.get("current_csv_path"),
    )
