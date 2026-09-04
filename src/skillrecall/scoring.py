"""Routing simulation: which skill wins a task when all are installed together.

A `Router` scores every candidate's resident text against a task. The
lexical scorer is BM25 over unigrams and adjacent bigrams with an
inverted index, so a task touches only the documents that share a term.
An optional dense scorer contributes meaning-level similarity; the two are
combined per task after standardising each, so neither dominates by scale.

Guard clauses are handled as rules rather than words. A sentence such as
"Not for pricing pages; use clarity-pricing" is stripped from the indexed
text (otherwise it would pull pricing tasks toward this skill) and applied
as a yield: when a task matches the guard's condition, this skill steps
aside for the named one.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .tasks import TEMPLATES, Task
from .text import STOP, name_words, terms, tokens

_TEMPLATE_WORDS = frozenset(t for tpl in TEMPLATES for t in tokens(tpl.replace("{s}", "")))

GUARD_CUES = re.compile(
    r"\b(?:not for|instead|use [a-z0-9-]+ (?:for|when|if)|see |prefer|defer to|rather than|"
    r"belongs to|handled by|is for|covered by|leave .* to|hand(?:s|ed)? off to|for .*, use)\b",
    re.I,
)
_SENT_SPLIT = re.compile(r"(?<=[.;!?])\s+")
_CUE_WORDS = frozenset("not for instead use when if see prefer defer rather than belongs handled covered leave hand off to is".split())

YIELD_FACTOR = 0.25  # a guard hit hands the task to the named skill
SOFT_YIELD_FACTOR = 0.6  # a bare pointer only yields when the named skill is competitive


@dataclass(slots=True)
class Guard:
    target: str  # normalised neighbour name the guard points at
    condition: frozenset[str]  # task terms that trigger the yield; empty means bare pointer


@dataclass(slots=True)
class Doc:
    id: str
    name: str
    text: str  # resident text as the host sees it
    origin: str = ""
    installs: int = 0
    guards: list[Guard] = field(default_factory=list)
    indexed_text: str = ""  # resident text with guard sentences removed

    def __post_init__(self) -> None:
        if not self.indexed_text:
            self.indexed_text = self.text


def normalise_name(name: str) -> str:
    return "-".join(name_words(name))


def parse_guards(description: str, known: dict[str, int]) -> tuple[list[Guard], str]:
    """Extract guard clauses that name a known neighbour; return them and the stripped text."""
    guards: list[Guard] = []
    kept: list[str] = []
    for sent in _SENT_SPLIT.split(description):
        low = sent.lower()
        hit = None
        for nm in known:
            if nm and (nm in low or nm.replace("-", " ") in low):
                hit = nm
                break
        if hit is not None and GUARD_CUES.search(sent):
            cond = frozenset(
                t
                for t in tokens(sent)
                if t not in _CUE_WORDS and t not in hit.split("-") and t not in STOP
            )
            guards.append(Guard(hit, cond))
        else:
            kept.append(sent)
    return guards, " ".join(kept)


class DenseScorer(Protocol):
    def similarities(self, query: str, texts: Sequence[str]) -> list[float]: ...


class Index:
    """BM25 over unigrams and bigrams with an inverted index."""

    __slots__ = ("n", "k1", "b", "dl", "avgdl", "idf", "postings", "term_counts")

    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.n = len(texts)
        self.k1, self.b = k1, b
        self.term_counts: list[Counter[str]] = [Counter(terms(t)) for t in texts]
        self.dl = [sum(c.values()) for c in self.term_counts]
        self.avgdl = (sum(self.dl) / self.n) if self.n else 1.0
        df: Counter[str] = Counter()
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, c in enumerate(self.term_counts):
            for t, tf in c.items():
                df[t] += 1
                postings[t].append((i, tf))
        n = self.n
        self.idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.postings = postings

    def scores(self, query_terms: set[str]) -> list[float]:
        out = [0.0] * self.n
        k1, b, avgdl, dl = self.k1, self.b, self.avgdl, self.dl
        idf, postings = self.idf, self.postings
        for t in query_terms:
            w = idf.get(t)
            if not w:
                continue
            for d, tf in postings[t]:
                out[d] += w * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl[d] / avgdl))
        return out

    def contributions(self, query_terms: set[str], d: int) -> dict[str, float]:
        """Per-term share of document `d`'s score for a query."""
        c = self.term_counts[d]
        out: dict[str, float] = {}
        k1, b = self.k1, self.b
        norm = 1.0 - b + b * self.dl[d] / self.avgdl
        for t in query_terms:
            tf = c.get(t)
            w = self.idf.get(t)
            if tf and w:
                out[t] = w * tf * (k1 + 1.0) / (tf + k1 * norm)
        return out


def _standardise(values: list[float]) -> list[float]:
    n = len(values)
    if n < 2:
        return [0.0] * n
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd == 0.0:
        return [0.0] * n
    return [(v - mean) / sd for v in values]


class Router:
    """Score tasks against a fixed set of docs, applying guards and optional dense similarity."""

    def __init__(self, docs: Sequence[Doc], dense: DenseScorer | None = None) -> None:
        self.docs = list(docs)
        self.index = Index([d.indexed_text for d in self.docs])
        self.dense = dense
        self.by_name = {normalise_name(d.name): i for i, d in enumerate(self.docs)}
        self._guarded = [i for i, d in enumerate(self.docs) if d.guards]

    def score(self, task_text: str) -> list[float]:
        qt = set(terms(task_text))
        s = self.index.scores(qt)
        if self.dense is not None:
            dense = self.dense.similarities(task_text, [d.text for d in self.docs])
            zs, zd = _standardise(s), _standardise(dense)
            s = [(a + c) / 2.0 for a, c in zip(zs, zd)]
        if self._guarded:
            q_tokens = set(tokens(task_text))
            for i in self._guarded:
                for g in self.docs[i].guards:
                    j = self.by_name.get(g.target)
                    if g.condition and (g.condition & q_tokens):
                        s[i] = s[i] * YIELD_FACTOR if s[i] > 0 else s[i] - 1.0
                    elif j is not None and s[j] >= 0.8 * s[i] and s[j] > 0:
                        s[i] = s[i] * SOFT_YIELD_FACTOR if s[i] > 0 else s[i] - 0.5
        return s

    def winner(self, scores: list[float]) -> int:
        best, bi = -math.inf, -1
        for i, v in enumerate(scores):
            if v > best:
                best, bi = v, i
        return bi if best > 0 or self.dense is not None else -1

    def rank_of(self, scores: list[float], i: int) -> int:
        target = scores[i]
        if target <= 0 and self.dense is None:
            return len(scores)
        return 1 + sum(1 for v in scores if v > target)


@dataclass(slots=True)
class Outcome:
    own_hits: list[bool]
    own_ranks: list[int]
    own_winners: list[int]
    adv_hits: list[bool]
    adv_owner: list[int]
    comp_hits: list[bool]
    no_match: int = 0  # own tasks that matched nothing at all


def evaluate(router: Router, self_idx: int, own: Sequence[Task], adversarial: Sequence[Task], composition: Sequence[Task], top_k: int = 3) -> Outcome:
    own_hits: list[bool] = []
    own_ranks: list[int] = []
    own_winners: list[int] = []
    no_match = 0
    for t in own:
        s = router.score(t.text)
        w = router.winner(s)
        if w == -1:
            no_match += 1
        own_hits.append(w == self_idx)
        own_ranks.append(router.rank_of(s, self_idx))
        own_winners.append(w)
    adv_hits: list[bool] = []
    adv_owner: list[int] = []
    for t in adversarial:
        s = router.score(t.text)
        adv_hits.append(router.winner(s) == self_idx)
        adv_owner.append(t.owner)
    comp_hits: list[bool] = []
    for t in composition:
        s = router.score(t.text)
        comp_hits.append(router.rank_of(s, self_idx) <= top_k)
    return Outcome(own_hits, own_ranks, own_winners, adv_hits, adv_owner, comp_hits, no_match)


def attribution(router: Router, self_idx: int, own: Sequence[Task], k: int = 6) -> tuple[list[str], list[str]]:
    """Terms in the resident text that earn the wins, and task terms the text lacks."""
    earned: Counter[str] = Counter()
    demand: Counter[str] = Counter()
    have = set(router.index.term_counts[self_idx])
    for t in own:
        qt = set(terms(t.text))
        for term, w in router.index.contributions(qt, self_idx).items():
            earned[term] += w
        for term in tokens(t.text):
            if term not in have and term not in _TEMPLATE_WORDS:
                demand[term] += 1
    carrying = [t for t, _ in earned.most_common(k)]
    n_docs = max(1, router.index.n)
    max_df = max(1, int(0.15 * n_docs))
    missing = [
        t
        for t, c in demand.most_common(k * 6)
        if c >= 2 and len(t) >= 4 and len(router.index.postings.get(t, ())) <= max_df
    ][:k]
    return carrying, missing


def build_doc(doc_id: str, name: str, description: str, known: dict[str, int], origin: str = "", installs: int = 0) -> Doc:
    """Make a Doc for a resident text, parsing guards against the known neighbour names."""
    guards, stripped = parse_guards(description, known)
    text = f"{name}: {description}"
    indexed = f"{name}: {stripped}" if guards else text
    return Doc(id=doc_id, name=name, text=text, origin=origin, installs=installs, guards=guards, indexed_text=indexed)
