"""The competition: every other skill the author's skill will sit next to.

Candidates come from three places and are tagged with their origin so the
report can say where a competitor lives:

* `installed`  - skill roots on this machine (the author's real environment)
* `local`      - directories the caller points at explicitly
* `catalog`    - a public catalog search (see `catalog.py`)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .skill import Skill, load_skill

INSTALLED_ROOTS = (
    "~/.claude/skills",
    "~/.agents/skills",
    "~/.codex/skills",
    "~/.cursor/skills",
    "~/.kimi/skills",
    "~/.config/opencode/skills",
)


@dataclass(slots=True)
class Candidate:
    id: str
    name: str
    description: str
    origin: str
    source: str = ""
    installs: int = 0
    body: str = ""
    url: str = ""
    skill: Skill | None = field(default=None, repr=False)

    @property
    def resident_text(self) -> str:
        return f"{self.name}: {self.description}"


def candidates_from_dir(root: str | os.PathLike[str], origin: str = "local") -> list[Candidate]:
    """Every immediate child directory with a SKILL.md, symlinks resolved and deduplicated."""
    r = Path(root).expanduser()
    if not r.is_dir():
        return []
    seen: set[str] = set()
    out: list[Candidate] = []
    for child in sorted(r.iterdir()):
        md = child / "SKILL.md"
        if not md.is_file():
            continue
        real = str(md.resolve())
        if real in seen:
            continue
        seen.add(real)
        try:
            sk = load_skill(child)
        except Exception:
            continue
        if not sk.description:
            continue
        out.append(
            Candidate(
                id=f"{origin}:{child.name}",
                name=sk.name,
                description=sk.description,
                origin=origin,
                source=str(child),
                body=sk.body,
                url=str(child),
                skill=sk,
            )
        )
    return out


def candidates_from_paths(paths: Iterable[str | os.PathLike[str]], origin: str) -> list[Candidate]:
    """Candidates from explicit skill directories (used for siblings in a collection)."""
    out: list[Candidate] = []
    for path in paths:
        d = Path(path)
        try:
            sk = load_skill(d)
        except Exception:
            continue
        if not sk.description:
            continue
        out.append(Candidate(id=f"{origin}:{d.name}", name=sk.name, description=sk.description, origin=origin, source=str(d), body=sk.body, url=str(d), skill=sk))
    return out


def installed_candidates(roots: Iterable[str] = INSTALLED_ROOTS) -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[str] = set()
    for root in roots:
        for c in candidates_from_dir(root, origin="installed"):
            real = str(Path(c.source).resolve())
            if real in seen:
                continue
            seen.add(real)
            out.append(c)
    return out


def dedupe_against(skill: Skill, cands: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    """Drop candidates that are the author's own skill (same path, or same name and description).

    Returns (kept, duplicates). Duplicates are reported separately because a
    second install of the same skill is itself a finding.
    """
    own_real = str(Path(skill.path).resolve())
    kept: list[Candidate] = []
    dups: list[Candidate] = []
    for c in cands:
        same_path = False
        if c.origin in ("installed", "local"):
            try:
                same_path = str(Path(c.source).resolve()) == own_real
            except OSError:
                same_path = False
        if same_path:
            continue
        if c.name == skill.name and c.description.strip() == skill.description.strip():
            dups.append(c)
            continue
        kept.append(c)
    return kept, dups


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    p = Path(base) / "skillrecall"
    p.mkdir(parents=True, exist_ok=True)
    return p


def state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    p = Path(base) / "skillrecall"
    p.mkdir(parents=True, exist_ok=True)
    return p
