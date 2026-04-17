"""
A logging filter that scrubs key-shaped strings from log records.

Defensive backstop: code should never log keys in the first place, but this
catches accidental leaks during development.
"""

from __future__ import annotations

import logging
import re

# Rough shapes of common API keys. Over-inclusive on purpose.
_KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),          # OpenAI-style
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),      # Anthropic-style
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),         # Google API key
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.=]{20,}"),
]

REDACTED = "***redacted***"


class SecretScrubber(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        redacted = msg
        for pat in _KEY_PATTERNS:
            redacted = pat.sub(REDACTED, redacted)
        if redacted != msg:
            record.msg = redacted
            record.args = ()
        return True


def install() -> None:
    """Attach the scrubber to the root logger."""
    root = logging.getLogger()
    if not any(isinstance(f, SecretScrubber) for f in root.filters):
        root.addFilter(SecretScrubber())
