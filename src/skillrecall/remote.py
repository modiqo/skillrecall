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

import difflib
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
    if not raw.startswith(("http://", "https://")) and raw.startswith(("github.com/", "skills.sh/", "www.skills.sh/")):
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
            branch, path, catalog_id = "HEAD", "", ""
            if host == "github.com" and len(parts) >= 4 and parts[2] in ("tree", "blob"):
                branch = parts[3]
                path = "/".join(parts[4:])
            elif host == "raw.githubusercontent.com" and len(parts) >= 3:
                branch = parts[2]
                path = "/".join(parts[3:])
            elif host == "github.com" and len(parts) >= 3:
                # github.com/owner/repo/<skill-or-path>: try it as a path, then as a skill name.
                path = "/".join(parts[2:])
                if len(parts) == 3:
                    catalog_id = f"{owner}/{repo}/{parts[2]}"
            if path.endswith("SKILL.md"):
                path = path[: -len("SKILL.md")].rstrip("/")
            return RemoteRef(
                f"{owner}/{repo}",
                path,
                branch,
                f"https://github.com/{owner}/{repo}" + (f"/tree/{branch}/{path}" if path else ""),
                catalog_id,
            )
        raise ValueError(f"unsupported host in {ref}")
    m = _SHORT.match(raw)
    if not m:
        raise ValueError(f"not a recognised skill reference: {ref}")
    owner, repo, rest = m.group(1), m.group(2), m.group(3) or ""
    if rest and "/" not in rest:
        # owner/repo/name: a path if such a directory exists, else a skill name.
        return RemoteRef(f"{owner}/{repo}", rest, "HEAD", f"https://skills.sh/{owner}/{repo}/{rest}", f"{owner}/{repo}/{rest}")
    return RemoteRef(f"{owner}/{repo}", rest, "HEAD", f"https://github.com/{owner}/{repo}" + (f"/tree/HEAD/{rest}" if rest else ""), "")


def _tree(source: str, ref: str, cache: _Cache, timeout: float) -> list[str]:
    key = f"{source}@{ref}"
    tree = cache.get("tree", key, FILE_TTL)
    if tree is None:
        data = _http(TREE_URL_REF.format(source=source, ref=ref), timeout, accept="application/vnd.github+json")
        if data is None:
            raise RuntimeError(_tree_failure(source, timeout))
        try:
            tree = [t["path"] for t in json.loads(data).get("tree", []) if t.get("type") == "blob"]
        except (ValueError, KeyError, TypeError) as e:
            raise RuntimeError(f"unexpected tree listing for {source}") from e
        cache.put("tree", key, tree)
    return tree


def _tree_failure(source: str, timeout: float) -> str:
    """Explain why a tree listing failed, naming the rate limit when that is the cause."""
    data = _http("https://api.github.com/rate_limit", timeout, accept="application/vnd.github+json")
    try:
        core = json.loads(data)["resources"]["core"] if data else None
    except (ValueError, KeyError, TypeError):
        core = None
    if core and core.get("remaining", 1) == 0:
        reset = time.strftime("%H:%M", time.localtime(core.get("reset", 0)))
        return (
            f"GitHub API rate limit exhausted ({core.get('limit')} requests/hour without a token); it resets at {reset}.\n"
            "Set GITHUB_TOKEN, or run `gh auth login`, to get 5,000 requests/hour."
        )
    return f"could not list files of {source} (network problem or private repository)"


def _locate(r: RemoteRef, tree: list[str]) -> str:
    """Directory inside the repo that holds SKILL.md for this reference."""
    if r.path and f"{r.path}/SKILL.md" in tree:
        return r.path
    if r.path and not r.catalog_id:
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
        raise FileNotFoundError(guidance(r.source, skill_names(tree), missing=skill))
    if "SKILL.md" in tree:
        return ""
    candidates = [p for p in tree if p.endswith("SKILL.md")]
    if len(candidates) == 1:
        return candidates[0][: -len("SKILL.md")].rstrip("/")
    if not candidates:
        raise FileNotFoundError(f"no SKILL.md in {r.source}")
    raise ValueError(guidance(r.source, skill_names(tree)))


def skill_names(tree: list[str]) -> list[str]:
    return sorted({d.rsplit("/", 1)[-1] for d in skill_dirs_in(tree) if d})


def guidance(source: str, names: list[str], missing: str = "", too_many: bool = False, local: bool = False) -> str:
    """A formatted message that shows exactly how to be specific about a skill."""
    names = sorted(names)
    lines: list[str] = []
    if missing:
        close = difflib.get_close_matches(missing, names, n=3, cutoff=0.5)
        lines.append(f"No skill named “{missing}” in {source}.")
        if close:
            lines.append(f"Did you mean: {', '.join(close)}")
        example = (close or names[:1] or ["<skill>"])[0]
    elif too_many:
        lines.append(f"{source} holds {len(names)} skills. That is a lot to assess at once, so choose.")
        example = names[0] if names else "<skill>"
    else:
        lines.append(f"{source} holds {len(names)} skills.")
        example = names[0] if names else "<skill>"
    second = next((n for n in names if n != example), example)
    lines.append("")
    lines.append("Pick one:")
    lines.append(f"  skillrecall assess {source}/{example}")
    if not local:
        lines.append(f"  skillrecall assess https://github.com/{source}/{example}")
    lines.append("")
    lines.append("Pick a few, and they compete against each other too:")
    lines.append(f"  skillrecall assess {source} --pick {example},{second}")
    lines.append("")
    if too_many:
        lines.append(f"Or assess all {len(names)} (several minutes, mostly waiting on the catalog):")
        lines.append(f"  skillrecall assess {source} --all")
    else:
        lines.append("Or assess all of them together, each against its siblings:")
        lines.append(f"  skillrecall assess {source}")
    lines.append("")
    if names:
        lines.append(f"Skills in {source}:")
        width = max(len(n) for n in names) + 2
        cols = max(1, min(4, 100 // width))
        for i in range(0, len(names), cols):
            lines.append("  " + "".join(n.ljust(width) for n in names[i : i + cols]).rstrip())
    return "\n".join(lines)


def skill_dirs_in(tree: list[str], under: str = "") -> list[str]:
    """Every directory holding a SKILL.md below `under`, hidden and vendored paths excluded."""
    prefix = f"{under}/" if under else ""
    out = []
    for p in tree:
        if not p.endswith("SKILL.md") or not p.startswith(prefix):
            continue
        parts = p.split("/")
        if any(seg.startswith(".") or seg == "node_modules" for seg in parts[:-1]):
            continue
        out.append(p[: -len("SKILL.md")].rstrip("/"))
    return sorted(out)


@dataclass(slots=True)
class RemoteCollection:
    """A repository or path holding several skills, listed but not yet downloaded."""

    ref: RemoteRef
    tree: list[str]
    dirs: list[str]  # repo-relative skill directories

    @property
    def names(self) -> list[str]:
        return [d.rsplit("/", 1)[-1] for d in self.dirs]

    def fetch(self, pick: list[str] | None = None, timeout: float = 12.0, workers: int = 8, cache: _Cache | None = None) -> list[Path]:
        cache = cache or _Cache()
        chosen = self.dirs
        if pick:
            wanted = {p.strip() for p in pick if p.strip()}
            chosen = [d for d in self.dirs if d.rsplit("/", 1)[-1] in wanted]
            missing = sorted(wanted - {d.rsplit("/", 1)[-1] for d in chosen})
            if missing:
                raise FileNotFoundError(guidance(self.ref.source, self.names, missing=missing[0]))
        with ThreadPoolExecutor(max_workers=max(1, workers // 2)) as ex:
            return list(ex.map(lambda d: _download(self.ref, self.tree, d, cache, timeout, 4), chosen))


def resolve(
    ref: str, timeout: float = 12.0, workers: int = 8, cache: _Cache | None = None
) -> tuple[str, Path | RemoteCollection, RemoteRef]:
    """Detect the shape of a remote reference.

    Returns ("skill", downloaded_dir, ref) for a single skill, or
    ("collection", RemoteCollection, ref) when the reference names a
    repository or path holding several skills. A collection is listed only;
    call `.fetch()` on the skills the caller decides to assess.
    """
    r = parse_ref(ref)
    cache = cache or _Cache()
    tree = _tree(r.source, r.ref, cache, timeout)
    try:
        skill_dir = _locate(r, tree)
        return "skill", _download(r, tree, skill_dir, cache, timeout, workers), r
    except ValueError as ambiguous:
        dirs = skill_dirs_in(tree, r.path if r.path and not r.catalog_id else "")
        if len(dirs) < 2:
            raise ambiguous
        return "collection", RemoteCollection(r, tree, dirs), r


def materialise(ref: str, timeout: float = 12.0, workers: int = 8, cache: _Cache | None = None) -> tuple[Path, RemoteRef]:
    """Download one skill's text files into the cache and return the local directory."""
    r = parse_ref(ref)
    cache = cache or _Cache()
    tree = _tree(r.source, r.ref, cache, timeout)
    skill_dir = _locate(r, tree)
    return _download(r, tree, skill_dir, cache, timeout, workers), r


def _download(r: RemoteRef, tree: list[str], skill_dir: str, cache: _Cache, timeout: float, workers: int) -> Path:
    prefix = f"{skill_dir}/" if skill_dir else ""
    wanted = [
        p
        for p in tree
        if p.startswith(prefix) and Path(p).suffix.lower() in _TEXT_EXT and not any(seg.startswith(".") for seg in p.split("/"))
    ]
    wanted.sort(key=lambda p: (0 if p == f"{prefix}SKILL.md" else 1, p))
    wanted = wanted[:MAX_FILES]
    if f"{prefix}SKILL.md" not in wanted:
        raise FileNotFoundError(f"no SKILL.md under {skill_dir or '/'} in {r.source}")

    dest = cache_dir() / "remote" / r.source.replace("/", "__") / r.ref / (skill_dir.replace("/", "__") or "_root")
    stamp = dest / ".fetched"
    if stamp.is_file() and time.time() - stamp.stat().st_mtime < FILE_TTL and (dest / "SKILL.md").is_file():
        return dest
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
    return dest


def local_shape(path: str) -> tuple[str, list[Path]]:
    """("skill", [dir]) when `path` is a skill, ("collection", [dirs...]) when it holds several."""
    p = Path(path).expanduser()
    if p.is_file():
        return "skill", [p.parent]
    if (p / "SKILL.md").is_file():
        return "skill", [p]
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(p):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "node_modules"]
        if "SKILL.md" in filenames:
            found.append(Path(dirpath))
            dirnames[:] = []
        elif Path(dirpath).relative_to(p).parts.__len__() >= 3:
            dirnames[:] = []
    if not found:
        raise FileNotFoundError(f"no SKILL.md under {p}")
    if len(found) == 1:
        return "skill", found
    return "collection", sorted(found)
