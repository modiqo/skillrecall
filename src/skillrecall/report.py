"""Render an assessment for people (plain language) or machines (JSON)."""

from __future__ import annotations

import json
from collections import Counter

from .assess import Assessment
from .stats import per_ten_phrase


def _fmt_installs(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M installs"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K installs"
    if n > 0:
        return f"{n} installs"
    return ""


def _label(n) -> str:
    if n.origin == "catalog" and n.source:
        return f"{n.name} ({n.source})"
    if n.origin == "sibling":
        return f"{n.name} (same repo)"
    return n.name


def _competition_line(a: Assessment) -> str:
    by = Counter(n.origin for n in a.neighbours)
    parts = []
    if by.get("sibling"):
        parts.append(f"{by['sibling']} in the same repo")
    if by.get("installed"):
        parts.append(f"{by['installed']} installed here")
    if by.get("local"):
        parts.append(f"{by['local']} local")
    if by.get("catalog"):
        parts.append(f"{by['catalog']} from the public catalog")
    total = len(a.neighbours)
    return f"{total} other skill{'s' if total != 1 else ''}" + (f" ({', '.join(parts)})" if parts else "")


def render_human(a: Assessment, detail: str = "simple", explain: bool = False) -> str:
    L: list[str] = []
    s = a.skill
    L.append(f"Skill: {s.name}")
    if a.options.source_url:
        L.append(f"Source: {a.options.source_url}")
    else:
        L.append(f"Path:  {s.path}")
    L.append(f"Competition: {_competition_line(a)}")
    if a.duplicates:
        L.append(f"Note: an identical copy of this skill also exists at: {', '.join(a.duplicates)}")
    if a.catalog_status.get("error"):
        L.append(f"Note: {a.catalog_status['error']}; results use local competition only.")
    L.append("")

    if a.movement:
        L.append("Since last run")
        for m in a.movement:
            L.append(f"  {m}")
        L.append("")

    L.append("How often you get picked")
    L.append(f"  When a task is yours, you are chosen {per_ten_phrase(a.recall.value)}.")
    L.append(f"  Confidence: {a.confidence} ({a.recall.n} sample tasks).")
    if a.weak_seeds:
        L.append("  The body has few example requests, so tasks were drawn from the description itself.")
        L.append("  Add a “When to use” section with real requests and rerun before trusting the advice below.")
    if a.composition.n:
        L.append(
            f"  In bigger requests that need several skills, you are among the top {a.options.top_k} {per_ten_phrase(a.composition.value)}."
        )
    if a.no_match_share >= 0.2:
        L.append(f"  {per_ten_phrase(a.no_match_share)} of your own tasks match no description at all, yours included.")
    L.append("")

    L.append("Who takes your tasks")
    stealers = [n for n in a.stealers if n.takes >= 0.05]
    if stealers:
        for n in stealers[:6]:
            extra = []
            if n.installs:
                extra.append(_fmt_installs(n.installs))
            if n.guarded:
                extra.append("you already hand off to it")
            if n.guards_me:
                extra.append("it hands off to you")
            L.append(f"  {_label(n):<52} takes {per_ten_phrase(n.takes):<14} {' · '.join(extra)}".rstrip())
    else:
        L.append("  Nothing takes a meaningful share.")
    L.append("")

    L.append("Whose tasks you take")
    stolen = [n for n in a.stolen_from if n.taken >= 0.1]
    if stolen:
        for n in stolen[:6]:
            L.append(f"  {_label(n):<52} you answer {per_ten_phrase(n.taken)} of its tasks by mistake")
    else:
        L.append("  You leave other skills' tasks alone.")
    L.append("")

    L.append("Do these, in order")
    if a.edits:
        for i, e in enumerate(a.edits[:6], start=1):
            L.append(f"  {i}. {e.instruction}")
            L.append(f"     Expected: {e.verdict}.")
            if e.body_note:
                L.append("     Then add to the body:")
                for line in e.body_note.splitlines()[:6]:
                    L.append(f"       {line}")
        if a.applied_edits:
            L.append("")
            L.append("  Applied together, the header becomes:" if len(a.applied_edits) > 1 else "  Applied, the header becomes:")
            if a.suggested_name != s.name:
                L.append(f"    name: {a.suggested_name}")
            L.append(f"    description: {a.suggested_description}")
            r_now, r_new = a.recall.value, a.recall.value + a.suggested_recall_delta
            f_now, f_new = a.false_positives.value, a.false_positives.value + a.suggested_false_pos_delta
            size = (
                f"{abs(a.suggested_token_delta)} tokens {'shorter' if a.suggested_token_delta < 0 else 'longer'}"
                if a.suggested_token_delta
                else "same length"
            )
            L.append(
                f"  Expected: picked {per_ten_phrase(r_new)} (from {per_ten_phrase(r_now)}); mistaken pickups {per_ten_phrase(f_new)} (from {per_ten_phrase(f_now)}); {size}."
            )
    else:
        L.append("  No description change measured as an improvement. The structure notes below still apply.")
    L.append("")

    L.append("Structure")
    for f in a.findings:
        if f.severity == "ok":
            L.append(f"  {f.message}")
            continue
        mark = "fix" if f.severity == "fix" else "consider"
        L.append(f"  [{mark}] {f.message}")
        if f.action:
            L.append(f"        {f.action}")
    L.append("")
    L.append("Next")
    L.append("  Edit the description or body, then run the same command again. The report will open with what moved.")
    L.append("")

    if detail == "detailed":
        L.extend(_detailed(a))
    if explain:
        L.extend(_explain(a))
    return "\n".join(L).rstrip() + "\n"


def _detailed(a: Assessment) -> list[str]:
    L: list[str] = []
    s = a.skill
    L.append("Size")
    L.append(f"  Name + description: {s.resident_tokens} tokens ({s.token_label}); description alone {s.description_tokens}.")
    L.append(
        f"  Body: {s.lines} lines, about {s.body_tokens:,} tokens. Reference files: {len(s.reference_files)} ({s.reference_tokens:,} tokens)."
    )
    big = s.largest_reference
    if big:
        L.append(f"  Largest reference: {big.path} ({big.tokens:,} tokens).")
    L.append("")
    L.append("Words doing the work")
    L.append(f"  Carrying your wins: {', '.join(t.replace('_', ' ') for t in a.carrying_terms) or '(none)'}")
    L.append(f"  Asked for but absent: {', '.join(a.missing_terms) or '(none)'}")
    L.append("")
    L.append("All competitors")
    L.append(f"  {'name':<52} {'from':<10} {'takes':>6} {'taken':>6}  installs")
    for n in sorted(a.neighbours, key=lambda n: (-n.takes, -n.taken, -n.installs))[:40]:
        L.append(f"  {_label(n)[:52]:<52} {n.origin:<10} {n.takes:>6.2f} {n.taken:>6.2f}  {_fmt_installs(n.installs)}")
    if len(a.neighbours) > 40:
        L.append(f"  ... and {len(a.neighbours) - 40} more")
    L.append("")
    L.append("Every edit tried")
    for e in a.all_edits:
        r = f"{e.recall.delta:+.2f}" if e.recall else "  n/a"
        f = f"{e.false_pos.delta:+.2f}" if e.false_pos else "  n/a"
        L.append(
            f"  [{'keep' if e.accepted else 'skip'}] {e.instruction[:80]:<80} picked {r}  mistaken {f}  tokens {e.token_delta:+d}  {e.verdict}"
        )
    L.append("")
    L.append("Sample tasks used")
    for t in a.sample_tasks:
        L.append(f"  - {t}")
    L.append(f"  Sources: {', '.join(f'{k} {v}' for k, v in a.task_sources.items())}")
    L.append("")
    L.append("Run")
    L.append(f"  Scorers: {'; '.join(a.scorers)}")
    if a.reference:
        L.append(
            f"  Reference router picked you {per_ten_phrase(a.reference['recall'])} on {a.reference['n']} tasks and agreed with the local scorer {per_ten_phrase(a.reference['agreement'])}."
        )
    cs = a.catalog_status
    if cs.get("enabled"):
        L.append(
            f"  Catalog: {cs.get('found', 0)} found, {cs.get('fetched', 0)} with descriptions, {cs.get('unavailable', 0)} unavailable{', offline' if cs.get('offline') else ''}."
        )
    L.append(f"  Elapsed: {a.elapsed:.2f}s. Generated {a.generated_at}.")
    L.append("")
    return L


def _explain(a: Assessment) -> list[str]:
    r, f, c = a.recall, a.false_positives, a.composition
    return [
        "How this was measured",
        "  Tasks were sampled from the skill body (or the description when the body has few examples) and from",
        "  the bodies of the closest competitors. Each task was routed across every competitor's name and",
        "  description with a lexical scorer that treats hand-off sentences as rules, plus a dense scorer when",
        "  enabled. A task counts as picked when this skill scores highest. Every edit was applied to a copy",
        "  of the description and rescored on the same tasks; only edits with a measurable effect are shown.",
        f"  Picked: {r.value:.2f} [{r.low:.2f}, {r.high:.2f}] over {r.n} tasks. Mean reciprocal rank {a.mrr:.2f}.",
        f"  Mistaken pickups: {f.value:.2f} [{f.low:.2f}, {f.high:.2f}] over {f.n} competitor tasks.",
        f"  In top {a.options.top_k} on multi-skill requests: {c.value:.2f} [{c.low:.2f}, {c.high:.2f}] over {c.n} tasks.",
        "  Intervals are 95% bootstrap intervals. Seed and catalog snapshot make runs reproducible.",
        "",
    ]


def render_json(a: Assessment, detail: str = "simple") -> str:
    return json.dumps(a.as_dict(detail == "detailed"), indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Collections


def _first_step(a: Assessment) -> str:
    if a.edits:
        return a.edits[0].instruction
    fix = next((f for f in a.findings if f.severity == "fix"), None)
    if fix:
        return f"{fix.message} {fix.action}".strip()
    return "Nothing to change was measured."


def _loses_to(a: Assessment) -> str:
    st = [n for n in a.stealers if n.takes >= 0.05]
    if not st:
        return ""
    return ", ".join(f"{_label(n)} {per_ten_phrase(n.takes).replace('about ', '')}" for n in st[:3])


def _ref_for(c, a: Assessment) -> str:
    """The command that gives one skill's full report."""
    base = c.source
    if base.startswith("https://github.com/"):
        base = base[len("https://github.com/") :].split("/tree/")[0]
    return f"skillrecall assess {base}/{a.skill.name}"


WEAK_PICKUP = 0.5


def render_collection_human(c, detail: str = "simple", explain: bool = False) -> str:
    L: list[str] = []
    skills = sorted(c.skills, key=lambda a: (a.recall.value, -a.false_positives.value))
    n = len(skills)
    sample = skills[0].recall.n if skills else 0
    with_catalog = any(x.origin == "catalog" for a in skills for x in a.neighbours)
    against = f"its {n - 1} sibling{'s' if n - 1 != 1 else ''}" + (" and up to 40 skills from the public catalog" if with_catalog else "")

    L.append(f"Collection: {c.source}")
    L.append(f"{n} skills, each scored on {sample} sample requests against {against}. Took {c.elapsed:.0f}s.")
    if c.failures:
        L.append(f"Could not assess {len(c.failures)}: " + "; ".join(f"{name} ({err.splitlines()[0]})" for name, err in c.failures[:5]))
    L.append("")
    L.append("How to read this")
    L.append("  picked    how often a skill wins a request meant for it. 10 in 10 is best.")
    L.append("  mistaken  how often it wins a request meant for another skill. 0 in 10 is best.")
    L.append("  *         the body has no example requests, so requests were drawn from the description. Low confidence.")
    L.append("")

    weak = [a for a in skills if a.recall.value < WEAK_PICKUP]
    fine = [a for a in skills if a.recall.value >= WEAK_PICKUP]

    L.append(f"Needs attention: {len(weak)} skill{'s' if len(weak) != 1 else ''} picked under {int(WEAK_PICKUP * 10)} in 10")
    if not weak:
        L.append("  None. Every skill wins at least half of its own requests.")
    for a in weak:
        flag = " *" if a.weak_seeds else ""
        L.append(f"  {a.skill.name}{flag}")
        L.append(f"    picked {per_ten_phrase(a.recall.value)}, mistaken {per_ten_phrase(a.false_positives.value)}")
        loses = _loses_to(a)
        if loses:
            L.append(f"    loses to: {loses}")
        L.append(f"    do first: {_first_step(a)}")
        L.append(f"    full report: {_ref_for(c, a)}")
    L.append("")

    L.append(f"Doing fine: {len(fine)} skill{'s' if len(fine) != 1 else ''}")
    if fine:
        L.append(f"  {'skill':<32} {'picked':>8} {'mistaken':>9}  do first")
        for a in fine:
            flag = "*" if a.weak_seeds else " "
            step = _first_step(a)
            L.append(f"  {a.skill.name[:32]:<32} {a.recall.per_ten:>2} in 10  {a.false_positives.per_ten:>2} in 10 {flag} {step}")
    L.append("")

    pairs = [p for p in c.sibling_pairs if p.a_takes_b >= 0.1 or p.b_takes_a >= 0.1]
    L.append("Skills in this collection that take each other's requests")
    if pairs:
        for p in pairs[:10]:
            L.append(
                f"  {p.a} answers {per_ten_phrase(p.a_takes_b)} of {p.b}'s requests"
                + (f"; {p.b} answers {per_ten_phrase(p.b_takes_a)} of {p.a}'s" if p.b_takes_a >= 0.1 else "")
            )
        L.append("  Fix: give each of these a one-sentence hand-off naming the other, so the host knows which to pick.")
    else:
        L.append("  None above 1 in 10. They stay out of each other's way.")
    L.append("")

    L.append("What to do next")
    step = 1
    if weak:
        L.append(f"  {step}. Start with the weakest: {_ref_for(c, weak[0])}")
        L.append("     It shows who takes its requests, every edit that measurably helps, and the rewritten header.")
        step += 1
    starved = [a for a in skills if a.weak_seeds]
    if starved:
        L.append(f"  {step}. {len(starved)} of {n} bodies have no “When to use” section with example requests.")
        L.append(
            "     Adding three to eight real requests to each is the single change that helps most, and it makes these numbers trustworthy."
        )
        step += 1
    if pairs:
        L.append(f"  {step}. Add hand-off sentences for the {len(pairs)} overlapping pair{'s' if len(pairs) != 1 else ''} above.")
        step += 1
    L.append(f"  {step}. Edit, then run this command again. The report opens with what moved.")
    if detail == "detailed":
        for a in skills:
            L.append("")
            L.append("=" * 78)
            L.append(render_human(a, "detailed", explain).rstrip())
    return "\n".join(L).rstrip() + "\n"


def render_collection_json(c, detail: str = "simple") -> str:
    return json.dumps(c.as_dict(detail == "detailed"), indent=2, ensure_ascii=False) + "\n"
