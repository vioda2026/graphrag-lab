from __future__ import annotations

import re
from typing import Set

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_tokens(text: str) -> Set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def overlap_score(a: str, b: str) -> float:
    ta = normalize_tokens(a)
    tb = normalize_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
