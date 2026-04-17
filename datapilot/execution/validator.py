"""
AST-based validator for LLM-generated Python code.

Rejects code that uses dangerous primitives before it is ever executed.
This is a defence-in-depth layer: even with a restricted execution namespace,
we prefer to reject obviously dangerous code statically.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

# Modules the generated code may import.
ALLOWED_IMPORTS = frozenset(
    {
        "pandas", "pd",
        "numpy", "np",
        "matplotlib", "matplotlib.pyplot", "plt",
        "seaborn", "sns",
        "sklearn",
        "scipy", "scipy.stats",
        "statsmodels", "statsmodels.api",
        "xgboost",
        "lightgbm",
        "shap",
        "joblib",
        "math",
        "json",
        "re",
        "itertools",
        "collections",
        "typing",
    }
)

# Top-level names that, when called, are forbidden.
FORBIDDEN_CALLS = frozenset(
    {
        "eval", "exec", "compile", "__import__", "open",
        "input", "breakpoint", "globals", "locals", "vars",
    }
)

# Attribute chains that are forbidden anywhere in the code.
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "os.system", "os.popen", "os.execv", "os.execvp", "os.spawnl",
        "os.environ", "os.getenv", "os.putenv",
        "subprocess.Popen", "subprocess.call", "subprocess.run",
        "subprocess.check_call", "subprocess.check_output",
        "socket.socket", "socket.create_connection",
        "urllib.request.urlopen", "http.client.HTTPConnection",
        "requests.get", "requests.post", "requests.request",
        "pathlib.Path.write_bytes", "pathlib.Path.write_text",
        "pathlib.Path.unlink", "pathlib.Path.rmdir",
        "shutil.rmtree",
    }
)

# Modules that must never be imported, even if prefix-matched against
# ALLOWED_IMPORTS.
FORBIDDEN_MODULES = frozenset(
    {
        "os", "sys", "subprocess", "socket", "urllib", "urllib.request",
        "http", "http.client", "requests", "ctypes", "shutil", "pickle",
        "marshal", "importlib", "pathlib",
    }
)


@dataclass
class ValidationError:
    line: int
    col: int
    reason: str

    def __str__(self) -> str:
        return f"line {self.line}:{self.col}: {self.reason}"


def validate_code(source: str) -> list[ValidationError]:
    """
    Return a list of violations. Empty list means the code is acceptable.
    A syntactically-invalid cell produces a single error entry.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [ValidationError(e.lineno or 0, e.offset or 0, f"syntax error: {e.msg}")]

    errors: list[ValidationError] = []
    for node in ast.walk(tree):
        errors.extend(_check_node(node))
    return errors


def _check_node(node: ast.AST) -> list[ValidationError]:
    errs: list[ValidationError] = []
    line = getattr(node, "lineno", 0)
    col = getattr(node, "col_offset", 0)

    if isinstance(node, ast.Import):
        for alias in node.names:
            if _is_forbidden_import(alias.name):
                errs.append(ValidationError(line, col, f"forbidden import: {alias.name}"))
    elif isinstance(node, ast.ImportFrom):
        if node.module and _is_forbidden_import(node.module):
            errs.append(ValidationError(line, col, f"forbidden import: {node.module}"))
    elif isinstance(node, ast.Call):
        errs.extend(_check_call(node))
    elif isinstance(node, ast.Attribute):
        chain = _attr_chain(node)
        if chain in FORBIDDEN_ATTRIBUTES:
            errs.append(ValidationError(line, col, f"forbidden attribute access: {chain}"))

    return errs


def _check_call(node: ast.Call) -> list[ValidationError]:
    errs: list[ValidationError] = []
    line, col = node.lineno, node.col_offset
    if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
        errs.append(ValidationError(line, col, f"forbidden call: {node.func.id}"))
    if isinstance(node.func, ast.Attribute):
        chain = _attr_chain(node.func)
        if chain in FORBIDDEN_ATTRIBUTES:
            errs.append(ValidationError(line, col, f"forbidden call: {chain}"))
    return errs


def _attr_chain(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_forbidden_import(name: str) -> bool:
    if name in FORBIDDEN_MODULES:
        return True
    top = name.split(".", 1)[0]
    if top in FORBIDDEN_MODULES:
        return True
    # Only allow modules on the allow-list (or their submodules).
    if name in ALLOWED_IMPORTS:
        return False
    if top in ALLOWED_IMPORTS:
        return False
    return True
