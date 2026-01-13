"""Shared utilities for offline VLLM runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_completed_ids(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()
    completed: Set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("id")
            if sid is not None:
                completed.add(str(sid))
    return completed


def slice_rows(rows: List[Dict[str, Any]], offset: int = 0, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if limit is None:
        return rows[offset:]
    return rows[offset : offset + limit]


def filter_pending(rows: Iterable[Dict[str, Any]], completed_ids: Set[str]) -> List[Dict[str, Any]]:
    pending: List[Dict[str, Any]] = []
    for sample in rows:
        sid = sample.get("id")
        if sid is None:
            pending.append(sample)
            continue
        if str(sid) not in completed_ids:
            pending.append(sample)
    return pending


def open_output_file(path: Path, resume: bool):
    mode = "a" if resume and path.exists() else "w"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open(mode, encoding="utf-8")
