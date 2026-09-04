"""Resolve a skill reference that is not a local path.

Accepted forms:

* ``https://skills.sh/<owner>/<repo>/<skill>``
* ``https://github.com/<owner>/<repo>`` (a repository whose root, or a
  conventional skills directory, holds the skill)
* ``https://github.com/<owner>/<repo>/tree/<branch>/<path>`` or the
  ``blob`` form pointing at ``SKILL.md``
* ``<owner>/<repo>/<skill>`` and ``<owner>/<repo>`` shorthand

The skill's text files are downloaded once into the cache and loaded like a
local directory, so every measurement, including reference-file sizes,
works the same way. Binary files and anything over the size limits are
skipped; only what an agent would read as text matters here.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .catalog import COMMON_PATHS, FILE_TTL, RAW_URL, _Cache, _http
from .corpus import cache_dir

_TEXT_EXT = frozenset({".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".csv", ".py", ".sh", ".ts", ".js", ".mjs", ".toml"})
_SEG = r"[A-Za-z0-9_.-]+"
_SHORT = re.compile(rf"^({_SEG})/({_SEG})(?:/(.+?))?/?$")
MAX_FILES = 60
MAX_FILE_BYTES = 2_000_000
TREE_URL_REF = "https://api.github.com/repos/{source}/git/trees/{ref}?recursive=1"


@dataclass(slots=True)
class RemoteRef:
    source: str  # owner/repo
    path: str  # directory inside the repo, "" for root
    ref: str  # branch, tag, or HEAD
    url: str  # canonical display URL
    catalog_id: str  # owner/repo/skill when known, else ""

    @property
    def key(self) -> str:
        return f"{self.source}@{self.ref}/{self.path}".rstrip("/")


def is_remote(ref: str) -> bool:
    if os.path.exists(os.path.expanduser(ref)):
        return False
    if ref.startswith(("http://", "https://", "github.com/", "skills.sh/", "www.skills.sh/")):
        return True
    return bool(_SHORT.match(ref)) and not ref.startswith((".", "/", "~"))


def parse_ref(ref: str) -> RemoteRef:
    raw = ref.strip()
    if not raw.startswith(("http://", "https://")):
        if raw.startswith(("github.com/", "skills.sh/", "www.skills.sh/")):
            raw = "https://" + raw
    if raw.startswith(("http://", "https://")):
        u = urlparse(raw)
        host = u.netloc.lower().removeprefix("www.")
        parts = [p for p in u.path.split("/") if p]
        if host == "skills.sh":
            if len(parts) < 3:
                raise ValueError(f"a skills.sh link needs owner/repo/skill: {ref}")
            owner, repo, skill = parts[0], parts[1], parts[2]
            return RemoteRef(f"{owner}/{repo}", "", "HEAD", f"https://skills.sh/{owner}/{repo}/{skill}", f"{owner}/{repo}/{skill}")
        if host in ("github.com", "raw.githubusercontent.com"):
            if len(parts) < 2:
                raise ValueError(f"a GitHub link needs owner/repo: {ref}")
            owner, repo = parts[0], parts[1].removesuffix(".git")
            branch, path = "HEAD", ""
            if host == "github.com" and len(parts) >= 4 and parts[2] in ("tree", "blob"):
                branch = parts[3]
                path = "/".join(parts[4:])
            elif host == "raw.githubusercontent.com" and len(parts) >= 3:
                branch = parts[2]
                path = "/".join(parts[3:])
            if path.endswith("SKILL.md"):
                path = path[: -len("SKILL.md")].rstrip("/")
            return RemoteRef(f"{owner}/{repo}", path, branch, f"https://github.com/{owner}/{repo}" + (f"/tree/{branch}/{path}" if path else ""), "")
        raise ValueError(f"unsupported host in {ref}")
    m = _SHORT.match(raw)
    if not m:
        raise ValueError(f"not a recognised skill reference: {ref}")
    owner, repo, rest = m.group(1), m.group(2), m.group(3) or ""
    if rest and "/" not in rest:
        return RemoteRef(f"{owner}/{repo}", "", "HEAD", f"https://skills.sh/{owner}/{repo}/{rest}", f"{owner}/{repo}/{rest}")
    return RemoteRef(f"{owner}/{repo}", rest, "HEAD", f"https://github.com/{owner}/{repo}" + (f"/tree/HEAD/{rest}" if rest else ""), "")


def _tree(source: str, ref: str, cache: _Cache, timeout: float) -> list[str]:
    key = f"{source}@{ref}"
    tree = cache.get("tree", key, FILE_TTL)
    if tree is None:
        data = _http(TREE_URL_REF.format(source=source, ref=ref), timeout, accept="application/vnd.github+json")
        if data is None:
            raise RuntimeError(f"could not list files of {source} (network, rate limit, or private repository)")
        try:
            tree = [t["path"] for t in json.loads(data).get("tree", []) if t.get("type") == "blob"]
        except (ValueError, KeyError, TypeError) as e:
            raise RuntimeError(f"unexpected tree listing for {source}") from e
        cache.put("tree", key, tree)
    return tree


def _locate(r: RemoteRef, tree: list[str]) -> str:
    """Directory inside the repo that holds SKILL.md for this reference."""
    if r.path:
        if f"{r.path}/SKILL.md" in tree:
            return r.path
        raise FileNotFoundError(f"no SKILL.md under {r.path} in {r.source}")
    skill = r.catalog_id.rsplit("/", 1)[-1] if r.catalog_id else ""
    if skill:
        for pattern in COMMON_PATHS:
            p = pattern.format(name=skill)
            if p in tree:
                return p[: -len("/SKILL.md")]
        for p in tree:
            if p.endswith(f"/{skill}/SKILL.md"):
                return p[: -len("/SKILL.md")]
        raise FileNotFoundError(f"skill {skill} not found in {r.source}")
    if "SKILL.md" in tree:
        return ""
    candidates = [p for p in tree if p.endswith("SKILL.md")]
    if len(candidates) == 1:
        return candidates[0][: -len("SKILL.md")].rstrip("/")
    if not candidates:
        raise FileNotFoundError(f"no SKILL.md in {r.source}")
    names = ", ".join(sorted(c[: -len("/SKILL.md")].rsplit("/", 1)[-1] for c in candidates)[:12])
    raise ValueError(f"{r.source} holds {len(candidates)} skills; name one, for example {r.source}/<skill>. Found: {names}")


def materialise(ref: str, timeout: float = 12.0, workers: int = 8, cache: _Cache | None = None) -> tuple[Path, RemoteRef]:
    """Download the skill's text files into the cache and return the local directory."""
    r = parse_ref(ref)
    cache = cache or _Cache()
    tree = _tree(r.source, r.ref, cache, timeout)
    skill_dir = _locate(r, tree)
    prefix = f"{skill_dir}/" if skill_dir else ""
    wanted = [p for p in tree if p.startswith(prefix) and Path(p).suffix.lower() in _TEXT_EXT and not any(seg.startswith(".") for seg in p.split("/"))]
    wanted.sort(key=lambda p: (0 if p == f"{prefix}SKILL.md" else 1, p))
    wanted = wanted[:MAX_FILES]
    if f"{prefix}SKILL.md" not in wanted:
        raise FileNotFoundError(f"no SKILL.md under {skill_dir or '/'} in {r.source}")

    dest = cache_dir() / "remote" / r.source.replace("/", "__") / r.ref / (skill_dir.replace("/", "__") or "_root")
    stamp = dest / ".fetched"
    if stamp.is_file() and time.time() - stamp.stat().st_mtime < FILE_TTL and (dest / "SKILL.md").is_file():
        return dest, r
    dest.mkdir(parents=True, exist_ok=True)

    def one(p: str) -> None:
        data = _http(RAW_URL.format(source=r.source, path=p).replace("/HEAD/", f"/{r.ref}/", 1), timeout)
        if data is None or len(data) > MAX_FILE_BYTES:
            return
        out = dest / p[len(prefix) :]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, wanted))
    if not (dest / "SKILL.md").is_file():
        raise RuntimeError(f"could not download SKILL.md from {r.source}")
    stamp.touch()
    return dest, r
