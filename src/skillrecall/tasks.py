"""Task sampling: the requests a skill is supposed to answer.

Recall is measured against tasks, never against other descriptions. Tasks
are drawn from the skill body (usage sections, bullets, examples, quoted
phrases, imperative sentences) so the description is never scored against
text copied from itself. When the body offers too little, the description
is used and the sample is flagged as weak. Real user requests, when
supplied, take precedence over everything generated.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .skill import split_sections
from .text import strip_markdown

_CUE = re.compile(
    r"when to use|use (?:this|it|when)|triggers?|use cases?|examples?|scenarios?|when this applies|"
    r"good for|typical (?:requests?|prompts?|asks?)|invoke|activate|prompts?",
    re.I,
)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*\S)\s*$", re.M)
_INSTRUCTION_SECTION = re.compile(
    r"procedure|steps?|workflow|process|how to|protocol|method|instructions|approach|caveat|limit|"
    r"pitfall|anti-?pattern|do not|never|warning|constraint|reference|resources?|output|format|notes?|"
    r"install|setup|requirements?|dependenc|configuration|architecture|implementation",
    re.I,
)
_QUOTED = re.compile(r"[\"“]([^\"”\n]{6,160})[\"”]")
_FENCE = re.compile(r"```.*?```", re.S)
_HTML = re.compile(r"<[^>]+>")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)
_IMPERATIVE = re.compile(
    r"^(?:audit|review|generate|create|build|write|check|find|analy[sz]e|convert|extract|summari[sz]e|"
    r"deploy|fix|run|test|scan|compare|measure|plan|draft|design|migrate|refactor|debug|optimi[sz]e|"
    r"search|fetch|parse|validate|format|translate|explain|list|monitor|track|schedule|install|"
    r"configure|set up|inspect|assess|evaluate|estimate|rewrite|improve|clean|render|export|import|"
    r"query|sync|publish|verify|diagnose|triage|document|profile|benchmark|score|rank|classify)\b",
    re.I,
)

MIN_WORDS, MAX_WORDS = 3, 40
WEAK_SEED_COUNT = 5

# How people actually phrase requests. The bare form appears twice so the
# sample is not dominated by prefixes.
TEMPLATES = (
    "{s}",
    "{s}",
    "help me {s}",
    "can you {s}",
    "I need to {s}",
    "how do I {s}",
    "please {s}",
    "{s} for my project",
)


@dataclass(slots=True)
class Task:
    text: str
    owner: int  # index of the skill the task belongs to
    source: str  # where the seed came from


def _clean(s: str) -> str:
    s = strip_markdown(_HTML.sub("", s))
    s = s.strip(" -:;,.\t")
    if s and s[0] in "\"'“”":
        s = s.strip("\"'“”")
    return s


def _ok(s: str) -> bool:
    n = len(s.split())
    return MIN_WORDS <= n <= MAX_WORDS and not s.lower().startswith(("note", "see ", "http"))


def seeds_from_body(body: str) -> list[str]:
    """Candidate task phrasings from a skill body, most reliable sources first."""
    body = _FENCE.sub("", body)
    body = _TABLE_ROW.sub("", body)
    sections = split_sections(body)
    cue_text: list[str] = []
    other_text: list[str] = []
    for sec in sections:
        if _CUE.search(sec.heading):
            cue_text.append(sec.text)
        elif sec.level == 0 or not _INSTRUCTION_SECTION.search(sec.heading):
            other_text.append(sec.text)
    seen: set[str] = set()
    out: list[str] = []

    def add(s: str) -> None:
        s = _clean(s)
        key = s.lower()
        if _ok(s) and key not in seen:
            seen.add(key)
            out.append(s)

    for chunk in cue_text:
        for m in _BULLET.finditer(chunk):
            add(m.group(1))
        for m in _QUOTED.finditer(chunk):
            add(m.group(1))
    for chunk in cue_text + other_text:
        for m in _QUOTED.finditer(chunk):
            add(m.group(1))
    for chunk in other_text:
        for m in _BULLET.finditer(chunk):
            item = m.group(1)
            if _IMPERATIVE.match(_clean(item)):
                add(item)
    for chunk in cue_text:
        for para in re.split(r"\n\s*\n", chunk):
            first = _clean(para.split(". ")[0])
            if _IMPERATIVE.match(first):
                add(first)
    return out


def seeds_from_description(description: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in _QUOTED.finditer(description):
        s = _clean(m.group(1))
        if _ok(s) and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    for sent in re.split(r"(?<=[.!?])\s+", strip_markdown(description)):
        s = _clean(sent)
        if _ok(s) and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def seeds_for(skill_body: str, description: str) -> tuple[list[str], bool]:
    """Seeds for a skill, plus whether they had to fall back to the description."""
    seeds = seeds_from_body(skill_body) if skill_body else []
    if len(seeds) >= WEAK_SEED_COUNT:
        return seeds, False
    extra = [s for s in seeds_from_description(description) if s not in seeds]
    return seeds + extra, True


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if s and s[0].isupper() and not s[:2].isupper() else s


def sample_tasks(seeds: list[str], n: int, owner: int, source: str, seed: int = 7) -> list[Task]:
    """Draw `n` tasks deterministically: every seed once, then templated variety."""
    if not seeds or n <= 0:
        return []
    rng = random.Random(seed * 1000003 + owner)
    out: list[Task] = []
    for s in seeds[:n]:
        out.append(Task(s, owner, source))
    while len(out) < n:
        s = rng.choice(seeds)
        t = rng.choice(TEMPLATES)
        text = t.format(s=_lower_first(s)) if t != "{s}" else s
        out.append(Task(text, owner, source))
    return out


def load_task_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def composition_tasks(own: list[Task], others: list[Task], n: int, owner: int, seed: int = 7) -> list[Task]:
    """Requests that need the author's skill plus another one.

    Pairs an own task with a task belonging to a different skill so the
    author's skill must surface in the top few, not only first.
    """
    if not own or not others or n <= 0:
        return []
    rng = random.Random(seed * 7919 + owner)
    out: list[Task] = []
    for _ in range(n):
        a = rng.choice(own)
        b = rng.choice(others)
        joiner = rng.choice([" and then ", ", then ", " and also ", "; also "])
        out.append(Task(f"{a.text}{joiner}{_lower_first(b.text)}", owner, f"composition:{b.owner}"))
    return out
