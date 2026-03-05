from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _to_jsonable(payload: Any) -> Any:
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, dict):
        return {k: _to_jsonable(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_to_jsonable(v) for v in payload]
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
