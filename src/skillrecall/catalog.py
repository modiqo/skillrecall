"""Public catalog lookup: find the skills an author will compete with.

The catalog exposes an unauthenticated search that ranks skills by meaning,
not just words, and reports install counts. It does not return descriptions,
so each neighbour's SKILL.md is fetched from its source repository. Both
steps are cached on disk, fetched in parallel, and fail soft: a neighbour
that cannot be fetched is counted and skipped, never fatal.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .corpus import Candidate, cache_dir
from .skill import load_skill_text

SEARCH_URL = os.environ.get("SKILLRECALL_SEARCH_URL", "https://skills.sh/api/search")
RAW_URL = "https://raw.githubusercontent.com/{source}/HEAD/{path}"
TREE_URL = "https://api.github.com/repos/{source}/git/trees/HEAD?recursive=1"
USER_AGENT = "skillrecall/0.1 (+https://github.com/modiqo/skillrecall)"

SEARCH_TTL = 60 * 60
FILE_TTL = 7 * 24 * 60 * 60
MAX_QUERY_CHARS = 400

# Where skill directories usually live inside a repository, tried in order
# before falling back to a full tree listing.
COMMON_PATHS = (
    "skills/{name}/SKILL.md",
    "{name}/SKILL.md",
    ".claude/skills/{name}/SKILL.md",
    ".agents/skills/{name}/SKILL.md",
    ".codex/skills/{name}/SKILL.md",
    "src/skills/{name}/SKILL.md",
    "packages/{name}/SKILL.md",
)

_SOURCE_OK = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


@dataclass(slots=True)
class CatalogStatus:
    queries: int = 0
    found: int = 0
    fetched: int = 0
    unavailable: int = 0
    offline: bool = False
    error: str = ""


class _Cache:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or cache_dir()

    def _path(self, kind: str, key: str) -> Path:
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        d = self.root / kind
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{h}.json"

    def get(self, kind: str, key: str, ttl: int):
        p = self._path(kind, key)
        try:
            if time.time() - p.stat().st_mtime > ttl:
                return None
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def put(self, kind: str, key: str, value) -> None:
        p = self._path(kind, key)
        tmp = p.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(value, f)
            os.replace(tmp, p)
        except OSError:
            pass


_GH_TOKEN: list[str | None] = []


def github_token() -> str | None:
    """GITHUB_TOKEN or GH_TOKEN, else the token the gh CLI holds, else None. Resolved once."""
    if _GH_TOKEN:
        return _GH_TOKEN[0]
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        try:
            import subprocess

            out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=5)
            token = out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            token = None
    _GH_TOKEN.append(token or None)
    return _GH_TOKEN[0]


def _http(url: str, timeout: float, accept: str = "*/*") -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    token = github_token() if "api.github.com" in url else None
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


class Catalog:
    """Search the public catalog and hydrate neighbours with their descriptions."""

    def __init__(self, timeout: float = 12.0, workers: int = 16, cache: _Cache | None = None, offline: bool = False) -> None:
        self.timeout = timeout
        self.workers = workers
        self.cache = cache or _Cache()
        self.offline = offline
        self.status = CatalogStatus(offline=offline)

    # -- search ------------------------------------------------------------

    def search(self, query: str, limit: int = 50) -> list[dict]:
        query = query.strip()[:MAX_QUERY_CHARS]
        if len(query) < 2:
            return []
        key = f"{query}|{limit}"
        cached = self.cache.get("search", key, SEARCH_TTL)
        if cached is not None:
            return cached
        if self.offline:
            return []
        self.status.queries += 1
        url = f"{SEARCH_URL}?{urllib.parse.urlencode({'q': query, 'limit': limit})}"
        data = _http(url, self.timeout, accept="application/json")
        if data is None:
            self.status.error = "catalog search unreachable"
            return []
        try:
            payload = json.loads(data)
        except ValueError:
            self.status.error = "catalog search returned non-JSON"
            return []
        skills = payload.get("skills") or payload.get("data") or []
        out = []
        for s in skills:
            sid = s.get("id") or ""
            source = s.get("source") or "/".join(sid.split("/")[:2])
            name = s.get("name") or s.get("skillId") or sid.rsplit("/", 1)[-1]
            if not sid or not _SOURCE_OK.match(source):
                continue
            out.append({"id": sid, "name": name, "source": source, "installs": int(s.get("installs") or 0)})
        self.cache.put("search", key, out)
        return out

    def neighbours(self, queries: list[str], limit: int = 50) -> list[dict]:
        """Union of several searches, keeping the first seen rank order."""
        seen: dict[str, dict] = {}
        for q in queries:
            for s in self.search(q, limit):
                if s["id"] not in seen:
                    seen[s["id"]] = s
        self.status.found = len(seen)
        return list(seen.values())

    # -- hydrate -----------------------------------------------------------

    def fetch_skill_md(self, source: str, name: str) -> str | None:
        key = f"{source}/{name}"
        cached = self.cache.get("skillmd", key, FILE_TTL)
        if cached is not None:
            return cached.get("text") or None
        if self.offline:
            return None
        text = None
        # A tree listing we already hold answers the path question without probing.
        known_tree = self.cache.get("tree", source, FILE_TTL)
        if known_tree is not None:
            path = self._path_in_tree(known_tree, name)
            if path:
                data = _http(RAW_URL.format(source=source, path=path), self.timeout)
                if data:
                    text = data.decode("utf-8", "replace")
        if text is None:
            for pattern in COMMON_PATHS[:3]:
                data = _http(RAW_URL.format(source=source, path=pattern.format(name=name)), self.timeout)
                if data and data.lstrip().startswith(b"---"):
                    text = data.decode("utf-8", "replace")
                    break
        if text is None:
            path = self._find_in_tree(source, name)
            if path:
                data = _http(RAW_URL.format(source=source, path=path), self.timeout)
                if data:
                    text = data.decode("utf-8", "replace")
        self.cache.put("skillmd", key, {"text": text or ""})
        return text

    @staticmethod
    def _path_in_tree(tree: list[str], name: str) -> str | None:
        suffix = f"/{name}/SKILL.md"
        for p in tree:
            if p.endswith(suffix) or p == f"{name}/SKILL.md":
                return p
        return None

    def _find_in_tree(self, source: str, name: str) -> str | None:
        tree = self.cache.get("tree", source, FILE_TTL)
        if tree is None:
            data = _http(TREE_URL.format(source=source), self.timeout, accept="application/vnd.github+json")
            if data is None:
                return None
            try:
                tree = [t["path"] for t in json.loads(data).get("tree", []) if t.get("type") == "blob"]
            except (ValueError, KeyError, TypeError):
                return None
            self.cache.put("tree", source, tree)
        return self._path_in_tree(tree, name)

    def hydrate(self, found: list[dict], max_neighbours: int = 40, budget: float = 45.0) -> list[Candidate]:
        """Turn search hits into candidates with descriptions, in parallel, within a time budget."""
        found = found[:max_neighbours]
        results: dict[str, Candidate | None] = {}
        started = time.monotonic()

        def one(hit: dict) -> tuple[str, Candidate | None]:
            if time.monotonic() - started > budget:
                return hit["id"], None  # out of time; counted as unavailable
            text = self.fetch_skill_md(hit["source"], hit["name"])
            if not text:
                return hit["id"], None
            sk = load_skill_text(hit["name"], text, origin=hit["id"])
            if not sk.description:
                return hit["id"], None
            return hit["id"], Candidate(
                id=hit["id"],
                name=sk.name,
                description=sk.description,
                origin="catalog",
                source=hit["source"],
                installs=hit.get("installs", 0),
                body=sk.body,
                url=f"https://skills.sh/{hit['id']}",
                skill=sk,
            )

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = [ex.submit(one, h) for h in found]
            for fut in as_completed(futures):
                try:
                    sid, cand = fut.result()
                except Exception:
                    continue
                results[sid] = cand
        out: list[Candidate] = []
        for h in found:
            c = results.get(h["id"])
            if c is None:
                self.status.unavailable += 1
            else:
                out.append(c)
        self.status.fetched = len(out)
        return out
