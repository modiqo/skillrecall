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
        text = a.edits[0].instruction
        return text if len(text) <= 60 else text[:57] + "..."
    fix = next((f for f in a.findings if f.severity == "fix"), None)
    if fix:
        return fix.message if len(fix.message) <= 60 else fix.message[:57] + "..."
    return "nothing measured"


def _biggest(a: Assessment) -> str:
    st = a.stealers
    if not st or st[0].takes < 0.05:
        return "-"
    n = st[0]
    return f"{_label(n)} takes {per_ten_phrase(n.takes).replace('about ', '')}"


def render_collection_human(c, detail: str = "simple", explain: bool = False) -> str:
    from .assess import Collection  # noqa: F401  (type only)

    L: list[str] = []
    n = len(c.skills)
    L.append(f"Collection: {c.source}")
    L.append(
        f"{n} skill{'s' if n != 1 else ''} assessed, each against its siblings"
        + (" and the public catalog" if any(x.origin == "catalog" for a in c.skills for x in a.neighbours) else "")
        + f", in {c.elapsed:.0f}s."
    )
    if c.failures:
        L.append(f"Could not assess {len(c.failures)}: " + ", ".join(f"{name} ({err})" for name, err in c.failures[:5]))
    L.append("")
    L.append("Weakest first")
    L.append(f"  {'skill':<30} {'picked':>8} {'mistaken':>9} {'top-3':>6}  {'biggest competitor':<44} first thing to do")
    for a in sorted(c.skills, key=lambda a: (a.recall.value, -a.false_positives.value)):
        picked = f"{a.recall.per_ten} in 10"
        mistaken = f"{a.false_positives.per_ten} in 10"
        top = f"{a.composition.per_ten}/10" if a.composition.n else "-"
        weak = "*" if a.weak_seeds else " "
        L.append(f"  {a.skill.name[:30]:<30} {picked:>8} {mistaken:>9} {top:>6} {weak} {_biggest(a)[:44]:<44} {_first_step(a)}")
    if any(a.weak_seeds for a in c.skills):
        L.append("  * body has few example requests; tasks came from the description, so treat that row as low confidence")
    L.append("")
    pairs = [p for p in c.sibling_pairs if p.a_takes_b >= 0.1 or p.b_takes_a >= 0.1]
    L.append("Pairs in this repo that take each other's tasks")
    if pairs:
        for p in pairs[:10]:
            L.append(
                f"  {p.a:<30} answers {per_ten_phrase(p.a_takes_b):<14} of {p.b}'s tasks; the reverse is {per_ten_phrase(p.b_takes_a)}"
            )
    else:
        L.append("  None above 1 in 10. The skills in this repo stay out of each other's way.")
    L.append("")
    L.append("For one skill's full report: skillrecall assess <this reference>/<skill>")
    if detail == "detailed":
        for a in sorted(c.skills, key=lambda a: a.recall.value):
            L.append("")
            L.append("=" * 78)
            L.append(render_human(a, "detailed", explain).rstrip())
    return "\n".join(L).rstrip() + "\n"


def render_collection_json(c, detail: str = "simple") -> str:
    return json.dumps(c.as_dict(detail == "detailed"), indent=2, ensure_ascii=False) + "\n"
