"""Counterfactual edits: try concrete changes and keep the ones that measurably help.

Every suggestion an author sees has already been applied to a copy of the
description and rescored against the same tasks as the baseline. Edits are
operations on (name, description) so accepted ones can be stacked into a
single suggested rewrite, which is rescored as a whole.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .scoring import Doc, Outcome, Router, build_doc, evaluate, normalise_name
from .skill import Skill
from .stats import PairedDelta, paired_delta
from .tasks import Task
from .text import TokenCounter, name_similarity, name_words, sentences

Apply = Callable[[str, str], tuple[str, str]]

_TRIGGER_SENT = re.compile(r"(trigger|use when the user says|use this whenever|examples? of requests?)", re.I)
_QUOTED = re.compile(r"[\"“]([^\"”\n]{4,160})[\"”]")
_OUTPUT_VERB = re.compile(r"\b(returns?|produces?|outputs?|generates?|gives?|reports?|writes?|creates?|delivers?)\b", re.I)

MIN_STEAL_SHARE = 0.05
TOKEN_PENALTY = 0.002  # score cost per added resident token
HARMLESS_SHORTENING = 8  # tokens saved that count as a win on their own


@dataclass(slots=True)
class Edit:
    kind: str
    instruction: str
    detail: str
    apply: Apply = field(repr=False)
    body_note: str = ""
    name: str = ""
    description: str = ""
    token_delta: int = 0
    recall: PairedDelta | None = None
    false_pos: PairedDelta | None = None
    composition: PairedDelta | None = None
    accepted: bool = False
    verdict: str = ""
    score: float = 0.0

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "instruction": self.instruction,
            "detail": self.detail,
            "body_note": self.body_note,
            "name": self.name,
            "description": self.description,
            "token_delta": self.token_delta,
            "recall": self.recall.as_dict() if self.recall else None,
            "false_positives": self.false_pos.as_dict() if self.false_pos else None,
            "composition": self.composition.as_dict() if self.composition else None,
            "accepted": self.accepted,
            "verdict": self.verdict,
        }


@dataclass(slots=True)
class Stealer:
    name: str
    share: float
    distinctive: list[str]


def _join_terms(ts: Sequence[str]) -> str:
    ts = [t.replace("_", " ") for t in ts]
    if not ts:
        return ""
    if len(ts) == 1:
        return ts[0]
    return ", ".join(ts[:-1]) + " and " + ts[-1]


def _remove_sentence(target: str) -> Apply:
    def f(name: str, desc: str) -> tuple[str, str]:
        ss = sentences(desc)
        if len(ss) <= 1 or target not in ss:
            return name, desc
        return name, " ".join(s for s in ss if s != target)

    return f


def _keep_first(k: int) -> Apply:
    def f(name: str, desc: str) -> tuple[str, str]:
        ss = sentences(desc)
        return name, " ".join(ss[:k]) if len(ss) > k else desc

    return f


def _drop_triggers() -> Apply:
    def f(name: str, desc: str) -> tuple[str, str]:
        kept = []
        for s in sentences(desc):
            quoted = _QUOTED.findall(s)
            if len(quoted) >= 3 or (_TRIGGER_SENT.search(s) and quoted):
                continue
            kept.append(s)
        return name, " ".join(kept) if kept else desc

    return f


def _append(sentence: str) -> Apply:
    def f(name: str, desc: str) -> tuple[str, str]:
        d = desc.rstrip()
        if d and d[-1] not in ".!?":
            d += "."
        return name, f"{d} {sentence}".strip()

    return f


def _rename(new_name: str) -> Apply:
    def f(name: str, desc: str) -> tuple[str, str]:
        return new_name, desc

    return f


def candidate_edits(
    skill: Skill,
    stealers: Sequence[Stealer],
    missing_terms: Sequence[str],
    carrying_terms: Sequence[str],
    neighbour_names: Sequence[str],
) -> list[Edit]:
    edits: list[Edit] = []
    desc = skill.description
    ss = sentences(desc)

    # 1. Each sentence removed on its own, except the opening sentence (it states the
    #    purpose) and any sentence that names the result: those are for people, and a
    #    lexical router cannot see their value.
    if len(ss) >= 2:
        for i, s in enumerate(ss):
            if i == 0 or _OUTPUT_VERB.search(s):
                continue
            short = s if len(s) <= 90 else s[:87] + "..."
            edits.append(Edit("remove_sentence", f"Remove sentence {i + 1}: “{short}”", s, _remove_sentence(s)))

    # 2. Trigger lists belong in the body.
    triggers = skill.trigger_phrases()
    if len(triggers) >= 3 or any(_TRIGGER_SENT.search(s) and _QUOTED.search(s) for s in ss):
        edits.append(
            Edit(
                "move_triggers",
                "Move the example phrases out of the description into a “When to use” section in the body",
                "; ".join(triggers[:8]),
                _drop_triggers(),
                body_note="## When to use\n" + "\n".join(f"- {t}" for t in triggers[:12]),
            )
        )

    # 3. A guard clause for every skill that takes a real share of the author's tasks.
    for st in stealers:
        if st.share < MIN_STEAL_SHARE or not st.distinctive:
            continue
        hint = _join_terms(st.distinctive[:2])
        sentence = f"Not for {hint} work; use {st.name} for that."
        edits.append(Edit("add_guard", f"Add a hand-off to {st.name}: “{sentence}”", sentence, _append(sentence)))

    # 4. Terms the tasks use that the description never says.
    if missing_terms:
        sentence = f"Also covers {_join_terms(missing_terms[:4])}."
        edits.append(
            Edit(
                "add_terms",
                f"Add the words people use when they ask for this: “{sentence}”",
                sentence,
                _append(sentence),
            )
        )

    # 5. A more distinctive name when the current one collides.
    norm = normalise_name(skill.name)
    collides = [n for n in neighbour_names if normalise_name(n) == norm or name_similarity(norm, normalise_name(n)) >= 0.85]
    if collides:
        have = set(name_words(skill.name))
        extra = next((t for t in carrying_terms if "_" not in t and t not in have and len(t) > 3), None)
        if extra:
            new_name = f"{norm}-{extra}"
            edits.append(
                Edit("rename", f"Rename to {new_name}; {collides[0]} already uses a near-identical name", new_name, _rename(new_name))
            )

    # 6. Shorter prefixes of the description.
    if len(ss) >= 3:
        for k in range(len(ss) - 1, 0, -1):
            edits.append(Edit("truncate", f"Keep only the first {k} sentence{'s' if k > 1 else ''}", " ".join(ss[:k]), _keep_first(k)))

    return edits


def _rescore(
    docs: list[Doc],
    self_idx: int,
    name: str,
    description: str,
    known: dict[str, int],
    own: Sequence[Task],
    adv: Sequence[Task],
    comp: Sequence[Task],
    dense,
) -> Outcome:
    new_docs = list(docs)
    old = docs[self_idx]
    new_docs[self_idx] = build_doc(old.id, name, description, known, old.origin, old.installs)
    router = Router(new_docs, dense)
    return evaluate(router, self_idx, own, adv, comp)


def evaluate_edits(
    edits: list[Edit],
    skill: Skill,
    docs: list[Doc],
    self_idx: int,
    known: dict[str, int],
    own: Sequence[Task],
    adv: Sequence[Task],
    comp: Sequence[Task],
    baseline: Outcome,
    counter: TokenCounter,
    dense=None,
    seed: int = 7,
    allow_shortening: bool = True,
) -> list[Edit]:
    base_tokens = counter.count(f"{skill.name}: {skill.description}")
    for e in edits:
        name, desc = e.apply(skill.name, skill.description)
        if desc.strip() == skill.description.strip() and name == skill.name:
            e.verdict = "no change"
            continue
        e.name, e.description = name, desc
        e.token_delta = counter.count(f"{name}: {desc}") - base_tokens
        out = _rescore(docs, self_idx, name, desc, known, own, adv, comp, dense)
        e.recall = paired_delta(baseline.own_hits, out.own_hits, seed)
        e.false_pos = paired_delta(baseline.adv_hits, out.adv_hits, seed)
        e.composition = paired_delta(baseline.comp_hits, out.comp_hits, seed)
        _judge(e, allow_shortening)
    ranked = [e for e in edits if e.accepted]
    ranked.sort(key=lambda e: -e.score)
    return ranked


def _judge(e: Edit, allow_shortening: bool = True) -> None:
    r, fp, c = e.recall, e.false_pos, e.composition
    assert r is not None and fp is not None and c is not None
    hurts = r.high < 0 or (fp.low > 0 and fp.delta >= max(r.delta, 0.0))
    score = r.delta - 0.5 * fp.delta + 0.25 * c.delta - TOKEN_PENALTY * max(0, e.token_delta)
    e.score = score
    if hurts:
        e.verdict = "would hurt"
        return
    gains = []
    if r.significant and r.delta > 0:
        gains.append("picked more often")
    if fp.significant and fp.delta < 0:
        gains.append("fewer mistaken pickups")
    if c.significant and c.delta > 0:
        gains.append("shows up more in bigger tasks")
    if gains:
        e.accepted = True
        e.verdict = ", ".join(gains)
        return
    if allow_shortening and e.token_delta <= -HARMLESS_SHORTENING and r.low >= -0.03 and fp.high <= 0.03:
        e.accepted = True
        e.verdict = f"shorter by {-e.token_delta} tokens, no loss"
        e.score += 0.01 * (-e.token_delta) / 10
        return
    e.verdict = "no measurable effect"


def compose(
    skill: Skill,
    ranked: Sequence[Edit],
    docs: list[Doc],
    self_idx: int,
    known: dict[str, int],
    own: Sequence[Task],
    adv: Sequence[Task],
    comp: Sequence[Task],
    baseline: Outcome,
    counter: TokenCounter,
    dense=None,
    seed: int = 7,
) -> tuple[str, str, list[Edit], PairedDelta, PairedDelta, PairedDelta, int]:
    """Stack accepted edits greedily, keeping each only if the stack still helps."""
    name, desc = skill.name, skill.description
    applied: list[Edit] = []
    best = baseline
    used_kinds: set[str] = set()
    removed_sentence = False
    for e in ranked:
        if e.kind == "truncate" and removed_sentence:
            continue
        if e.kind in ("rename", "move_triggers", "add_terms") and e.kind in used_kinds:
            continue
        n2, d2 = e.apply(name, desc)
        if (n2, d2) == (name, desc):
            continue
        out = _rescore(docs, self_idx, n2, d2, known, own, adv, comp, dense)
        r = paired_delta(best.own_hits, out.own_hits, seed)
        fp = paired_delta(best.adv_hits, out.adv_hits, seed)
        if r.high < 0 or (fp.low > 0 and r.delta <= 0):
            continue
        name, desc, best = n2, d2, out
        applied.append(e)
        used_kinds.add(e.kind)
        if e.kind == "remove_sentence":
            removed_sentence = True
    recall = paired_delta(baseline.own_hits, best.own_hits, seed)
    false_pos = paired_delta(baseline.adv_hits, best.adv_hits, seed)
    composition = paired_delta(baseline.comp_hits, best.comp_hits, seed)
    token_delta = counter.count(f"{name}: {desc}") - counter.count(f"{skill.name}: {skill.description}")
    return name, desc, applied, recall, false_pos, composition, token_delta


def states_output(description: str) -> bool:
    return bool(_OUTPUT_VERB.search(description))
