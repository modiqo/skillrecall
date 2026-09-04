"""Structural checks on the skill directory, with the sentence to add for each gap."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .skill import BODY_LINE_CAP, Skill

_WHEN = re.compile(r"when to use|use when|use this|triggers?|use cases?|when this applies|scope", re.I)
_PROCEDURE = re.compile(r"procedure|workflow|steps?|process|how to|protocol|method|instructions|approach", re.I)
_CAVEATS = re.compile(r"caveat|limit|pitfall|anti-?pattern|do not|don't|never|warning|constraint|out of scope|not for|when not", re.I)
_OUTPUT = re.compile(r"\b(returns?|produces?|outputs?|generates?|gives?|reports?|writes?|creates?|delivers?)\b", re.I)
_REF_POINTER = re.compile(r"references?/|\]\((?:\./)?[\w\-/]+\.(?:md|txt|json|yaml|yml)\)|scripts?/", re.I)

LARGE_REFERENCE_TOKENS = 40_000
THIN_DESCRIPTION_TOKENS = 15


@dataclass(slots=True)
class Finding:
    area: str
    severity: str  # "fix" | "consider" | "ok"
    message: str
    action: str = ""

    def as_dict(self) -> dict:
        return {"area": self.area, "severity": self.severity, "message": self.message, "action": self.action}


def _percentile(values: Sequence[int], x: int) -> float:
    if not values:
        return 0.5
    below = sum(1 for v in values if v < x)
    return below / len(values)


def structure_findings(skill: Skill, neighbour_description_tokens: Sequence[int]) -> list[Finding]:
    out: list[Finding] = []
    headings = [s.heading for s in skill.sections if s.heading]
    body_low = skill.body.lower()

    # Description budget against the competition.
    pct = _percentile(neighbour_description_tokens, skill.description_tokens)
    if neighbour_description_tokens and pct >= 0.9:
        n_all = len(neighbour_description_tokens)
        longer_than = sum(1 for v in neighbour_description_tokens if v < skill.description_tokens)
        who = "all" if longer_than == n_all else f"{longer_than} of the {n_all}"
        out.append(
            Finding(
                "description",
                "consider",
                f"Your description is longer than {who} skills it competes with. Every session pays for it whether or not the skill is used.",
                "Keep the sentences that name the task and the result; move examples and caveats into the body.",
            )
        )
    elif skill.description_tokens < THIN_DESCRIPTION_TOKENS:
        out.append(
            Finding(
                "description",
                "fix",
                "Your description is too short to separate this skill from its neighbours.",
                "Say what the skill does, what it takes in, and what it returns, in two or three sentences.",
            )
        )

    if not _OUTPUT.search(skill.description):
        out.append(
            Finding(
                "description",
                "consider",
                "The description never says what the skill produces.",
                "Add one clause naming the result, for example “returns a ranked list of fixes”, so multi-step requests can chain it.",
            )
        )

    if len(skill.trigger_phrases()) >= 3:
        out.append(
            Finding(
                "description",
                "consider",
                "The description carries a list of example phrases.",
                "Move them into a “When to use” section in the body; they are read only after selection.",
            )
        )

    # Body roles.
    if not any(_WHEN.search(h) for h in headings):
        out.append(Finding("body", "fix", "No “When to use” section.", "Add one with three to eight example requests and the situations that do not apply."))
    if not any(_PROCEDURE.search(h) for h in headings):
        out.append(Finding("body", "consider", "No procedure or steps section.", "Add the ordered steps an agent should follow once the skill is selected."))
    if not any(_CAVEATS.search(h) for h in headings) and not re.search(r"\b(do not|don't|never|not for)\b", body_low):
        out.append(Finding("body", "consider", "No caveats or limits section.", "State what the skill must not do and when to stop."))

    if skill.lines > BODY_LINE_CAP:
        out.append(
            Finding(
                "body",
                "fix",
                f"The body is {skill.lines} lines, over the {BODY_LINE_CAP}-line guideline.",
                "Move long material into files under references/ and point at them from the body.",
            )
        )

    # Deferred material.
    if skill.reference_files and not _REF_POINTER.search(skill.body):
        out.append(
            Finding(
                "references",
                "consider",
                f"{len(skill.reference_files)} reference file(s) ship with the skill but the body never points at them.",
                "Name each file and when to read it, or remove it.",
            )
        )
    big = skill.largest_reference
    if big and big.tokens >= LARGE_REFERENCE_TOKENS:
        out.append(
            Finding(
                "references",
                "fix",
                f"{big.path} is about {big.tokens:,} tokens; reading it whole would crowd out the task.",
                "Split it, or provide a script that looks up the needed part, and say so in the body.",
            )
        )

    if skill.script_count and "script" not in body_low:
        out.append(Finding("scripts", "consider", f"{skill.script_count} script file(s) ship with the skill but the body never mentions them.", "Say which script to run and when."))

    for w in skill.warnings:
        out.append(Finding("header", "fix", w, ""))

    if not out:
        out.append(Finding("structure", "ok", "Structure is in good shape.", ""))
    return out
