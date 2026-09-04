"""Per-skill history so a rerun can say what moved."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .corpus import state_dir


def _key(skill_path: str) -> str:
    return hashlib.sha1(str(Path(skill_path).resolve()).encode("utf-8")).hexdigest()[:16]


def history_dir(skill_path: str, root: Path | None = None) -> Path:
    d = (root or state_dir()) / _key(skill_path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(summary: dict, skill_path: str, root: Path | None = None) -> Path:
    d = history_dir(skill_path, root)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    p = d / f"{stamp}.json"
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    os.replace(tmp, p)
    return p


def history(skill_path: str, root: Path | None = None) -> list[dict]:
    d = history_dir(skill_path, root)
    out: list[dict] = []
    for p in sorted(d.glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                out.append(json.load(f))
        except (OSError, ValueError):
            continue
    return out


def latest(skill_path: str, root: Path | None = None) -> dict | None:
    h = history(skill_path, root)
    return h[-1] if h else None


def movement(prev: dict | None, cur: dict) -> list[str]:
    """Plain-language lines describing what changed since the previous run."""
    if not prev:
        return []
    lines: list[str] = []

    def per_ten(x: float) -> int:
        return int(round(x * 10))

    r0, r1 = prev.get("recall", 0.0), cur.get("recall", 0.0)
    if per_ten(r1) != per_ten(r0):
        word = "up" if r1 > r0 else "down"
        lines.append(f"Picked {per_ten(r1)} in 10 of your tasks, {word} from {per_ten(r0)} in 10.")
    else:
        lines.append(f"Picked {per_ten(r1)} in 10 of your tasks, unchanged.")
    f0, f1 = prev.get("false_positives", 0.0), cur.get("false_positives", 0.0)
    if per_ten(f1) != per_ten(f0):
        word = "down" if f1 < f0 else "up"
        lines.append(f"Mistaken pickups of other skills' tasks {word}: {per_ten(f1)} in 10, from {per_ten(f0)} in 10.")
    t0, t1 = prev.get("resident_tokens", 0), cur.get("resident_tokens", 0)
    if t0 and t1 != t0:
        pct = int(round(100 * (t1 - t0) / t0))
        lines.append(f"Description {'shorter' if t1 < t0 else 'longer'} by {abs(pct)}%.")
    n0, n1 = prev.get("name"), cur.get("name")
    if n0 and n1 and n0 != n1:
        lines.append(f"Renamed from {n0} to {n1}.")
    if len(lines) == 1 and "unchanged" in lines[0]:
        lines[0] = "Nothing moved since the last run."
    return lines
