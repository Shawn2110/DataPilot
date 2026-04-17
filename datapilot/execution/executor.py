"""
Execute validated code cells in a restricted namespace.

The executor:
  - Calls validate_code first; refuses to run if it fails.
  - Runs in a namespace pre-populated with the allow-listed libraries.
  - Pins file writes to the project directory via a wrapper around builtins.open.
  - Captures stdout.
  - Enforces a configurable timeout (best-effort on Python; uses signals on
    POSIX, a watchdog thread on Windows that cannot interrupt a tight loop).
"""

from __future__ import annotations

import builtins
import io
import signal
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datapilot.execution.validator import ValidationError, validate_code
from datapilot.storage.paths import safe_inside

DEFAULT_TIMEOUT_SECONDS = 60


@dataclass
class ExecResult:
    ok: bool
    stdout: str = ""
    error: str | None = None
    validation_errors: list[ValidationError] | None = None


class ExecutionTimeout(RuntimeError):
    pass


def execute(
    code: str,
    *,
    project_dir: Path,
    namespace: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> ExecResult:
    """Validate, then run. Returns ExecResult. Never raises for cell-level errors."""
    errors = validate_code(code)
    if errors:
        return ExecResult(ok=False, validation_errors=errors, error="validation failed")

    ns = _make_namespace(project_dir, namespace or {})
    buf = io.StringIO()

    try:
        with _time_limit(timeout), redirect_stdout(buf), redirect_stderr(buf):
            exec(compile(code, "<datapilot-cell>", "exec"), ns, ns)
    except ExecutionTimeout:
        return ExecResult(
            ok=False,
            stdout=buf.getvalue(),
            error=f"execution exceeded timeout of {timeout}s",
        )
    except Exception as e:
        return ExecResult(ok=False, stdout=buf.getvalue(), error=f"{type(e).__name__}: {e}")

    if namespace is not None:
        namespace.update({k: v for k, v in ns.items() if not k.startswith("__")})
    return ExecResult(ok=True, stdout=buf.getvalue())


# ---------------------------------------------------------------------------
# Namespace construction
# ---------------------------------------------------------------------------
def _make_namespace(project_dir: Path, base: dict[str, Any]) -> dict[str, Any]:
    ns: dict[str, Any] = dict(base)
    ns["__builtins__"] = _restricted_builtins(project_dir)
    _preload_allowed_libs(ns)
    return ns


def _restricted_builtins(project_dir: Path) -> dict[str, Any]:
    safe = {
        name: getattr(builtins, name)
        for name in dir(builtins)
        if not name.startswith("_")
        and name not in {"eval", "exec", "compile", "breakpoint", "input", "open"}
    }
    safe["open"] = _make_pinned_open(project_dir)
    return safe


def _make_pinned_open(project_dir: Path):
    """Wraps builtins.open so writes can only land inside project_dir."""
    def pinned_open(file, mode="r", *args, **kwargs):  # noqa: D401
        path = Path(file)
        writing = any(flag in mode for flag in ("w", "a", "x", "+"))
        if writing and not safe_inside(project_dir, path):
            raise PermissionError(
                f"refusing to write outside project directory: {path}"
            )
        return builtins.open(file, mode, *args, **kwargs)
    return pinned_open


def _preload_allowed_libs(ns: dict[str, Any]) -> None:
    """Pre-import the allow-listed libraries under their usual aliases."""
    try:
        import pandas as pd
        ns["pd"] = pd
        ns["pandas"] = pd
    except Exception:
        pass
    try:
        import numpy as np
        ns["np"] = np
        ns["numpy"] = np
    except Exception:
        pass
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        ns["plt"] = plt
        ns["matplotlib"] = matplotlib
    except Exception:
        pass
    try:
        import seaborn as sns
        ns["sns"] = sns
        ns["seaborn"] = sns
    except Exception:
        pass
    for mod_name in ("sklearn", "scipy", "statsmodels", "shap", "joblib"):
        try:
            ns[mod_name] = __import__(mod_name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Timeout: signal-based on POSIX, thread-based watchdog on Windows
# ---------------------------------------------------------------------------
class _time_limit:  # noqa: N801
    def __init__(self, seconds: int):
        self.seconds = seconds
        self._prev = None
        self._thread: threading.Timer | None = None
        self._posix = hasattr(signal, "SIGALRM")

    def __enter__(self):
        if self._posix:
            def _raise(signum, frame):  # noqa: ARG001
                raise ExecutionTimeout()
            self._prev = signal.signal(signal.SIGALRM, _raise)
            signal.alarm(self.seconds)
        else:
            self._thread = threading.Timer(self.seconds, _soft_kill)
            self._thread.daemon = True
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._posix:
            signal.alarm(0)
            if self._prev is not None:
                signal.signal(signal.SIGALRM, self._prev)
        elif self._thread is not None:
            self._thread.cancel()


def _soft_kill() -> None:
    """Best-effort interrupt on non-POSIX platforms."""
    try:
        import _thread

        _thread.interrupt_main()
    except Exception:
        sys.stderr.write("datapilot: timeout reached (soft kill)\n")
