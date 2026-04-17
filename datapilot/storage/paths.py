"""
Per-project directory helpers.

Every project lives under <storage_root>/projects/<slug>/. All directories are
created with mode 0700; session.json/notebook.ipynb with mode 0600.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    slug = slug.strip("._-")
    return slug or "untitled"


def ensure_project_dir(projects_root: Path, slug: str) -> Path:
    """
    Create the full subtree for a project under projects_root.
    Validates that the resolved path stays inside projects_root (no traversal).
    """
    projects_root = projects_root.expanduser().resolve()
    project_dir = (projects_root / slug).resolve()

    try:
        project_dir.relative_to(projects_root)
    except ValueError as e:
        raise ValueError(f"Project path escapes projects root: {project_dir}") from e

    for sub in ("", "data", "data/snapshots", "models", "charts", "export"):
        p = project_dir / sub if sub else project_dir
        p.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(p, 0o700)
        except (OSError, NotImplementedError):
            pass
    return project_dir


def session_path(project_dir: Path) -> Path:
    return project_dir / "session.json"


def notebook_path(project_dir: Path) -> Path:
    return project_dir / "notebook.ipynb"


def snapshot_path(project_dir: Path, cell_id: int) -> Path:
    return project_dir / "data" / "snapshots" / f"before_cell_{cell_id}.parquet"


def safe_inside(project_dir: Path, candidate: Path) -> bool:
    """True iff candidate resolves inside project_dir (symlinks resolved)."""
    try:
        candidate.resolve().relative_to(project_dir.resolve())
        return True
    except ValueError:
        return False
