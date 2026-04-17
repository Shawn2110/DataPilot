"""
PII detection and masking.

Used to scrub sensitive columns before any cloud-LLM egress, even in full-data
mode. Detection is pattern-based on column names — complementary, not a
substitute for data governance.
"""

from __future__ import annotations

import hashlib
import re

import pandas as pd

_PII_PATTERNS = [
    r"email",
    r"e[-_ ]?mail",
    r"\bssn\b",
    r"social[-_ ]?security",
    r"credit[-_ ]?card",
    r"\bcc[-_ ]?num",
    r"\bcard[-_ ]?no",
    r"phone",
    r"\bmobile\b",
    r"\btel\b",
    r"address",
    r"street",
    r"zip[-_ ]?code",
    r"postal",
    r"\bdob\b",
    r"date[-_ ]?of[-_ ]?birth",
    r"passport",
    r"license",
    r"\baccount\b",
    r"\biban\b",
    r"\btax[-_ ]?id\b",
    r"\bsin\b",
]
_PII_RE = re.compile("|".join(_PII_PATTERNS), re.IGNORECASE)


def is_pii_column(name: str) -> bool:
    return bool(_PII_RE.search(name))


def pii_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_pii_column(str(c))]


def mask_pii_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with PII columns replaced by a stable salted hash.
    Preserves row count + column set so downstream stats are still meaningful.
    """
    if df is None or df.empty:
        return df
    cols = pii_columns(df)
    if not cols:
        return df
    out = df.copy()
    for col in cols:
        out[col] = out[col].astype(str).map(_stable_hash)
    return out


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]
