"""Load a skill directory and measure the parts of it that affect selection.

A skill is a directory with a `SKILL.md` whose YAML header carries `name` and
`description`. Only those two fields are visible to the host when it decides
which skill to use; the body is read after selection, and files under the
directory are read only when the body points at them. The measurements here
follow that split so every number maps to a moment in the host's behaviour.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .text import TokenCounter, sentences, strip_markdown

_FRONT = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.S)
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$", re.M)
_TEXT_EXT = frozenset({".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".csv"})
_SCRIPT_EXT = frozenset({".py", ".sh", ".ts", ".js", ".mjs", ".rb", ".bash", ".zsh", ".go", ".rs"})
_QUOTED = re.compile(r"[\"“]([^\"”\n]{4,160})[\"”]")

BODY_LINE_CAP = 500


@dataclass(slots=True)
class ReferenceFile:
    path: str
    bytes: int
    tokens: int


@dataclass(slots=True)
class Section:
    level: int
    heading: str
    text: str


@dataclass(slots=True)
class Skill:
    """A parsed skill with the measurements the assessor needs."""

    path: str
    name: str
    description: str
    frontmatter: dict[str, str]
    body: str
    lines: int
    sections: list[Section]
    reference_files: list[ReferenceFile]
    script_count: int
    resident_tokens: int = 0
    description_tokens: int = 0
    body_tokens: int = 0
    reference_tokens: int = 0
    token_label: str = "estimated"
    warnings: list[str] = field(default_factory=list)

    @property
    def resident_text(self) -> str:
        return f"{self.name}: {self.description}"

    @property
    def largest_reference(self) -> ReferenceFile | None:
        return max(self.reference_files, key=lambda r: r.tokens, default=None)

    def section_matching(self, pattern: re.Pattern[str]) -> Section | None:
        for s in self.sections:
            if pattern.search(s.heading):
                return s
        return None

    def trigger_phrases(self) -> list[str]:
        """Quoted phrases in the description, which authors use as trigger lists."""
        return [m.group(1).strip() for m in _QUOTED.finditer(self.description)]


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse the small YAML subset skill headers use.

    Supports `key: value`, quoted values, folded (`>`) and literal (`|`) block
    scalars, and indented continuation lines. Nested mappings are flattened to
    `parent.child`. Anything else is kept as raw text under its key.
    """
    m = _FRONT.match(text)
    if not m:
        return {}, text
    fm: dict[str, str] = {}
    key: str | None = None
    block_indent = -1
    parents: list[tuple[int, str]] = []
    for raw in m.group(1).split("\n"):
        if not raw.strip():
            if key is not None and block_indent >= 0:
                fm[key] += "\n"
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if key is not None and block_indent >= 0 and indent >= block_indent:
            fm[key] = (fm[key] + " " + line).strip() if fm[key] else line
            continue
        block_indent = -1
        mk = re.match(r"^([A-Za-z0-9_-]+):(?:\s+(.*))?$", line)
        if mk and not line.startswith("-"):
            while parents and parents[-1][0] >= indent:
                parents.pop()
            name = mk.group(1)
            full = ".".join([p[1] for p in parents] + [name])
            val = (mk.group(2) or "").strip()
            if val in (">", "|", ">-", "|-"):
                key = full
                fm[key] = ""
                block_indent = indent + 1
                continue
            if val == "":
                parents.append((indent, name))
                key = None
                continue
            fm[full] = _unquote(val)
            key = full
            continue
        if key is not None and indent > 0:
            fm[key] = (fm[key] + " " + line).strip()
    for k, v in fm.items():
        fm[k] = re.sub(r"[ \t]+", " ", v).strip()
    return fm, text[m.end() :]


def _unquote(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    return v


def split_sections(body: str) -> list[Section]:
    marks = list(_HEADING.finditer(body))
    out: list[Section] = []
    if not marks:
        return [Section(0, "", body)]
    if marks[0].start() > 0:
        out.append(Section(0, "", body[: marks[0].start()]))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        out.append(Section(len(m.group(1)), m.group(2).strip(), body[m.end() : end]))
    return out


def load_skill(path: str | os.PathLike[str], counter: TokenCounter | None = None) -> Skill:
    """Load a skill from a directory or a SKILL.md path."""
    p = Path(path).expanduser()
    if p.is_dir():
        md_path = p / "SKILL.md"
    else:
        md_path = p
        p = p.parent
    if not md_path.is_file():
        raise FileNotFoundError(f"no SKILL.md at {md_path}")
    counter = counter or TokenCounter()
    raw = md_path.read_text(encoding="utf-8", errors="replace")
    fm, body = parse_frontmatter(raw)
    warnings: list[str] = []
    name = fm.get("name") or p.name
    if not fm.get("name"):
        warnings.append("header has no name; using the directory name")
    description = fm.get("description", "")
    if not description:
        warnings.append("header has no description; the host has nothing to route on")

    refs: list[ReferenceFile] = []
    scripts = 0
    root = p.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "node_modules"]
        for f in filenames:
            if f.startswith("."):
                continue
            fp = Path(dirpath) / f
            if fp.resolve() == md_path.resolve():
                continue
            ext = fp.suffix.lower()
            if ext in _TEXT_EXT:
                try:
                    data = fp.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                refs.append(ReferenceFile(str(fp.relative_to(root)), len(data.encode("utf-8", "replace")), counter.count(data)))
            elif ext in _SCRIPT_EXT:
                scripts += 1
    refs.sort(key=lambda r: -r.tokens)

    skill = Skill(
        path=str(root),
        name=name,
        description=description,
        frontmatter=fm,
        body=body,
        lines=raw.count("\n") + (0 if raw.endswith("\n") else 1),
        sections=split_sections(body),
        reference_files=refs,
        script_count=scripts,
        warnings=warnings,
    )
    skill.token_label = counter.label
    skill.resident_tokens = counter.count(skill.resident_text)
    skill.description_tokens = counter.count(description)
    skill.body_tokens = counter.count(raw)
    skill.reference_tokens = sum(r.tokens for r in refs)
    return skill


def load_skill_text(name_hint: str, raw: str, origin: str = "") -> Skill:
    """Build a lightweight Skill from SKILL.md text fetched elsewhere."""
    fm, body = parse_frontmatter(raw)
    return Skill(
        path=origin,
        name=fm.get("name") or name_hint,
        description=fm.get("description", ""),
        frontmatter=fm,
        body=body,
        lines=raw.count("\n") + 1,
        sections=split_sections(body),
        reference_files=[],
        script_count=0,
    )


def description_sentences(skill: Skill) -> list[str]:
    return sentences(strip_markdown(skill.description))
