"""Orchestration: from a skill directory to an assessment with ranked advice."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path

from . import snapshots
from .corpus import Candidate, candidates_from_dir, candidates_from_paths, dedupe_against, installed_candidates
from .edits import Edit, Stealer, candidate_edits, compose, evaluate_edits
from .remote import is_remote, materialise
from .scoring import Doc, Router, attribution, build_doc, evaluate, normalise_name
from .skill import Skill, load_skill
from .stats import Rate, confidence_word, rate
from .structure import Finding, structure_findings
from .tasks import Task, composition_tasks, load_task_lines, sample_tasks, seeds_for
from .text import TokenCounter, name_similarity, strip_markdown, tokens


@dataclass(slots=True)
class Options:
    skill_path: str
    corpus_dirs: list[str] = field(default_factory=list)
    include_installed: bool = False
    catalog: bool = True
    offline: bool = False
    neighbours: int = 40
    tasks: int = 100
    adversarial_per_neighbour: int = 15
    max_adversarial_neighbours: int = 25
    composition: int = 50
    tasks_file: str | None = None
    seed: int = 7
    dense: bool = False
    dense_model: str | None = None
    reference: bool = False
    reference_model: str = "claude-opus-5"
    reference_n: int = 30
    save_snapshot: bool = True
    state_root: Path | None = None
    top_k: int = 3
    timeout: float = 12.0
    source_url: str = ""  # filled in when skill_path was a remote reference
    sibling_dirs: list[str] = field(default_factory=list)  # other skills shipped with this one
    catalog_id: str = ""  # this skill's own catalog id, excluded from its competition
    progress: Callable[[str], None] | None = field(default=None, repr=False, compare=False)  # stage updates


@dataclass(slots=True)
class Neighbour:
    name: str
    origin: str
    source: str
    installs: int
    url: str
    description: str
    takes: float  # share of the author's tasks this neighbour wins
    taken: float  # share of this neighbour's tasks the author wins
    taken_n: int
    guarded: bool  # the author's description points at it
    guards_me: bool  # its description points at the author
    name_overlap: float

    def as_dict(self, detail: bool) -> dict:
        d = {
            "name": self.name,
            "origin": self.origin,
            "source": self.source,
            "installs": self.installs,
            "takes_share": round(self.takes, 4),
            "taken_share": round(self.taken, 4),
            "hand_off_present": self.guarded,
            "points_at_you": self.guards_me,
        }
        if detail:
            d.update(
                {"url": self.url, "description": self.description, "taken_n": self.taken_n, "name_overlap": round(self.name_overlap, 3)}
            )
        return d


@dataclass(slots=True)
class Assessment:
    skill: Skill
    options: Options
    recall: Rate
    mrr: float
    false_positives: Rate
    composition: Rate
    no_match_share: float
    weak_seeds: bool
    task_sources: dict[str, int]
    neighbours: list[Neighbour]
    duplicates: list[str]
    carrying_terms: list[str]
    missing_terms: list[str]
    edits: list[Edit]
    all_edits: list[Edit]
    suggested_name: str
    suggested_description: str
    suggested_recall_delta: float
    suggested_false_pos_delta: float
    suggested_composition_delta: float
    suggested_token_delta: int
    applied_edits: list[Edit]
    findings: list[Finding]
    scorers: list[str]
    catalog_status: dict
    reference: dict | None
    movement: list[str]
    previous_run: str | None
    sample_tasks: list[str]
    elapsed: float
    generated_at: str

    # -- convenience -------------------------------------------------------

    @property
    def confidence(self) -> str:
        word = confidence_word(self.recall.n)
        if self.weak_seeds and word in ("high", "moderate"):
            return "low"
        return word

    @property
    def stealers(self) -> list[Neighbour]:
        return sorted((n for n in self.neighbours if n.takes > 0), key=lambda n: -n.takes)

    @property
    def stolen_from(self) -> list[Neighbour]:
        return sorted((n for n in self.neighbours if n.taken > 0), key=lambda n: -n.taken)

    def summary(self) -> dict:
        """The compact record kept between runs."""
        return {
            "generated_at": self.generated_at,
            "name": self.skill.name,
            "path": self.skill.path,
            "recall": round(self.recall.value, 4),
            "false_positives": round(self.false_positives.value, 4),
            "composition": round(self.composition.value, 4),
            "resident_tokens": self.skill.resident_tokens,
            "description_tokens": self.skill.description_tokens,
            "body_lines": self.skill.lines,
            "neighbours": len(self.neighbours),
            "tasks": self.recall.n,
        }

    def as_dict(self, detail: bool = False) -> dict:
        d = {
            "schema": "skillrecall/v1",
            "generated_at": self.generated_at,
            "skill": {
                "name": self.skill.name,
                "path": self.skill.path,
                "source": self.options.source_url,
                "description": self.skill.description,
                "resident_tokens": self.skill.resident_tokens,
                "description_tokens": self.skill.description_tokens,
                "body_lines": self.skill.lines,
                "body_tokens": self.skill.body_tokens,
                "reference_files": len(self.skill.reference_files),
                "reference_tokens": self.skill.reference_tokens,
                "token_count": self.skill.token_label,
            },
            "competition": {
                "total": len(self.neighbours),
                "by_origin": dict(Counter(n.origin for n in self.neighbours)),
                "duplicates": self.duplicates,
                "catalog": self.catalog_status,
            },
            "pickup": {
                "recall": self.recall.as_dict(),
                "false_positives": self.false_positives.as_dict(),
                "composition_top_k": self.composition.as_dict(),
                "no_match_share": round(self.no_match_share, 4),
                "confidence": self.confidence,
                "weak_task_sample": self.weak_seeds,
                "mrr": round(self.mrr, 4),
            },
            "takes_your_tasks": [n.as_dict(detail) for n in self.stealers[:10]],
            "you_take_theirs": [n.as_dict(detail) for n in self.stolen_from[:10]],
            "edits": [e.as_dict() for e in self.edits],
            "suggested": {
                "name": self.suggested_name,
                "description": self.suggested_description,
                "recall_delta": round(self.suggested_recall_delta, 4),
                "false_positives_delta": round(self.suggested_false_pos_delta, 4),
                "composition_delta": round(self.suggested_composition_delta, 4),
                "token_delta": self.suggested_token_delta,
                "applied": [e.instruction for e in self.applied_edits],
            },
            "structure": [f.as_dict() for f in self.findings],
            "since_last_run": self.movement,
            "previous_run": self.previous_run,
        }
        if detail:
            d["terms"] = {"carrying": self.carrying_terms, "missing": self.missing_terms}
            d["neighbours"] = [n.as_dict(True) for n in self.neighbours]
            d["all_edits"] = [e.as_dict() for e in self.all_edits]
            d["sample_tasks"] = self.sample_tasks
            d["task_sources"] = self.task_sources
            d["scorers"] = self.scorers
            d["reference"] = self.reference
            d["elapsed_seconds"] = round(self.elapsed, 3)
            d["options"] = {
                f.name: (str(v) if isinstance(v := getattr(self.options, f.name), Path) else v)
                for f in fields(self.options)
                if f.name != "progress"
            }
        return d


# ---------------------------------------------------------------------------


def _gather(
    skill: Skill, opts: Options, self_id: str = "", note: Callable[[str], None] = lambda s: None
) -> tuple[list[Candidate], list[Candidate], dict]:
    cands: list[Candidate] = []
    cands.extend(candidates_from_paths(opts.sibling_dirs, origin="sibling"))
    for d in opts.corpus_dirs:
        cands.extend(candidates_from_dir(d, origin="local"))
    if opts.include_installed:
        cands.extend(installed_candidates())
    status: dict = {"enabled": bool(opts.catalog), "offline": opts.offline}
    if opts.catalog:
        from .catalog import Catalog

        cat = Catalog(timeout=opts.timeout, offline=opts.offline)
        desc = strip_markdown(skill.description)
        first = desc.split(". ")[0]
        queries = [desc, f"{' '.join(skill.name.split('-'))}: {first}"]
        note("searching the catalog for competitors")
        found = [h for h in cat.neighbours(queries, limit=min(200, max(20, opts.neighbours))) if h["id"] != self_id]
        note(f"fetching descriptions of {min(len(found), opts.neighbours)} competitors")
        cands.extend(cat.hydrate(found, max_neighbours=opts.neighbours))
        status.update({f.name: getattr(cat.status, f.name) for f in fields(cat.status)})
    # Merge: one candidate per (name, description) text, first origin wins.
    seen: set[tuple[str, str]] = set()
    merged: list[Candidate] = []
    for c in cands:
        key = (normalise_name(c.name), c.description.strip())
        if key in seen:
            continue
        seen.add(key)
        merged.append(c)
    kept, dups = dedupe_against(skill, merged)
    return kept, dups, status


def _distinctive(cand: Candidate, own_terms: set[str], k: int = 3) -> list[str]:
    c = Counter(t for t in tokens(f"{cand.name} {cand.description}") if t not in own_terms and len(t) > 3)
    return [t for t, _ in c.most_common(k)]


def assess(opts: Options) -> Assessment:
    t0 = time.perf_counter()
    note = opts.progress or (lambda s: None)
    counter = TokenCounter()
    self_id = opts.catalog_id
    if is_remote(opts.skill_path):
        note("fetching the skill")
        local, ref = materialise(opts.skill_path, timeout=opts.timeout)
        opts.source_url = ref.url
        self_id = ref.catalog_id
        skill = load_skill(local, counter)
    else:
        note("reading the skill")
        skill = load_skill(opts.skill_path, counter)
    cands, dups, cat_status = _gather(skill, opts, self_id, note)

    # Docs: the author's skill is index 0.
    known: dict[str, int] = {normalise_name(skill.name): 0}
    for i, c in enumerate(cands, start=1):
        known.setdefault(normalise_name(c.name), i)
    docs: list[Doc] = [build_doc("self", skill.name, skill.description, known, "self")]
    docs.extend(build_doc(c.id, c.name, c.description, known, c.origin, c.installs) for c in cands)

    dense = None
    scorers = ["lexical (BM25, unigrams + bigrams, guard-aware)"]
    if opts.dense:
        from .dense import load_dense

        dense = load_dense(opts.dense_model)
        scorers.append(f"dense ({dense.label})")

    # Own tasks.
    note(f"sampling requests against {len(cands)} competitors")
    seeds, weak = seeds_for(skill.body, skill.description)
    own: list[Task] = []
    sources: Counter[str] = Counter()
    if opts.tasks_file:
        for line in load_task_lines(opts.tasks_file):
            own.append(Task(line, 0, "provided"))
        sources["provided"] = len(own)
    if len(own) < opts.tasks:
        gen = sample_tasks(seeds, opts.tasks - len(own), 0, "description" if weak else "body", opts.seed)
        own.extend(gen)
        sources["description" if weak else "body"] += len(gen)

    # Adversarial tasks from the closest neighbours.
    probe = Router(docs)
    close = probe.score(f"{skill.name} {strip_markdown(skill.description)}")
    order = sorted(range(1, len(docs)), key=lambda i: -close[i])[: opts.max_adversarial_neighbours]
    adv: list[Task] = []
    for i in order:
        c = cands[i - 1]
        s2, _ = seeds_for(c.body, c.description)
        adv.extend(sample_tasks(s2, opts.adversarial_per_neighbour, i, "neighbour", opts.seed))
    sources["neighbour"] = len(adv)
    comp = composition_tasks(own, adv, opts.composition, 0, opts.seed)

    note(f"routing {len(own) + len(adv) + len(comp)} requests")
    router = Router(docs, dense)
    base = evaluate(router, 0, own, adv, comp, opts.top_k)

    recall = rate(base.own_hits, opts.seed)
    fp = rate(base.adv_hits, opts.seed)
    composition = rate(base.comp_hits, opts.seed)
    mrr = (sum(1.0 / r for r in base.own_ranks) / len(base.own_ranks)) if base.own_ranks else 0.0
    no_match_share = (base.no_match / len(own)) if own else 0.0

    # Who takes what.
    win_counts = Counter(w for w in base.own_winners if w > 0)
    taken_hits: Counter[int] = Counter()
    taken_n: Counter[int] = Counter()
    for hit, owner in zip(base.adv_hits, base.adv_owner, strict=False):
        taken_n[owner] += 1
        taken_hits[owner] += int(hit)
    own_norm = normalise_name(skill.name)
    neighbours: list[Neighbour] = []
    guard_targets = {g.target for g in docs[0].guards}
    for i, c in enumerate(cands, start=1):
        nn = normalise_name(c.name)
        neighbours.append(
            Neighbour(
                name=c.name,
                origin=c.origin,
                source=c.source,
                installs=c.installs,
                url=c.url,
                description=c.description,
                takes=(win_counts[i] / len(own)) if own else 0.0,
                taken=(taken_hits[i] / taken_n[i]) if taken_n[i] else 0.0,
                taken_n=taken_n[i],
                guarded=nn in guard_targets,
                guards_me=any(g.target == own_norm for g in docs[i].guards),
                name_overlap=name_similarity(own_norm, nn),
            )
        )

    carrying, missing = attribution(router, 0, own)

    # Edits.
    own_terms = set(tokens(skill.resident_text))
    stealers = [
        Stealer(c.name, win_counts[i] / len(own), _distinctive(c, own_terms)) for i, c in enumerate(cands, start=1) if own and win_counts[i]
    ]
    stealers.sort(key=lambda s: -s.share)
    all_edits = candidate_edits(skill, stealers[:5], missing, carrying, [c.name for c in cands])
    note(f"trying {len(all_edits)} edits")
    weak_sample = weak and not opts.tasks_file
    ranked = evaluate_edits(
        all_edits, skill, docs, 0, known, own, adv, comp, base, counter, dense, opts.seed, allow_shortening=not weak_sample
    )
    s_name, s_desc, applied, r_d, fp_d, c_d, tok_d = compose(skill, ranked, docs, 0, known, own, adv, comp, base, counter, dense, opts.seed)

    note("checking structure")
    findings = structure_findings(skill, [counter.count(c.description) for c in cands])

    reference = None
    if opts.reference:
        from .router import reference_check

        note(f"asking {opts.reference_model} to route {opts.reference_n} requests")
        reference = reference_check(router, 0, own, opts.reference_n, opts.reference_model, opts.seed).as_dict()
        scorers.append(f"reference router ({opts.reference_model}, {reference['n']} tasks)")

    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prev = snapshots.latest(skill.path, opts.state_root)
    a = Assessment(
        skill=skill,
        options=opts,
        recall=recall,
        mrr=mrr,
        false_positives=fp,
        composition=composition,
        no_match_share=no_match_share,
        weak_seeds=weak_sample,
        task_sources=dict(sources),
        neighbours=neighbours,
        duplicates=[f"{d.origin}: {d.source}" for d in dups],
        carrying_terms=carrying,
        missing_terms=missing,
        edits=applied,
        all_edits=all_edits,
        suggested_name=s_name,
        suggested_description=s_desc,
        suggested_recall_delta=r_d.delta,
        suggested_false_pos_delta=fp_d.delta,
        suggested_composition_delta=c_d.delta,
        suggested_token_delta=tok_d,
        applied_edits=applied,
        findings=findings,
        scorers=scorers,
        catalog_status=cat_status,
        reference=reference,
        movement=[],
        previous_run=prev.get("generated_at") if prev else None,
        sample_tasks=[t.text for t in own[:8]],
        elapsed=0.0,
        generated_at=generated_at,
    )
    a.movement = snapshots.movement(prev, a.summary())
    a.elapsed = time.perf_counter() - t0
    if opts.save_snapshot:
        snapshots.save(a.summary(), skill.path, opts.state_root)
    return a


# ---------------------------------------------------------------------------
# Collections: a repository or directory holding several skills.


@dataclass(slots=True)
class SiblingPair:
    a: str
    b: str
    a_takes_b: float  # share of b's tasks that a wins
    b_takes_a: float

    def as_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "a_takes_b": round(self.a_takes_b, 4), "b_takes_a": round(self.b_takes_a, 4)}


@dataclass(slots=True)
class Collection:
    source: str
    skills: list[Assessment]
    failures: list[tuple[str, str]]
    elapsed: float

    @property
    def sibling_pairs(self) -> list[SiblingPair]:
        takes: dict[tuple[str, str], float] = {}
        for a in self.skills:
            for n in a.neighbours:
                if n.origin == "sibling" and n.taken > 0:
                    takes[(a.skill.name, n.name)] = n.taken  # a wins n's tasks
        seen: set[frozenset[str]] = set()
        pairs: list[SiblingPair] = []
        for (x, y), share in takes.items():
            key = frozenset((x, y))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(SiblingPair(x, y, share, takes.get((y, x), 0.0)))
        pairs.sort(key=lambda p: -(p.a_takes_b + p.b_takes_a))
        return pairs

    def as_dict(self, detail: bool = False) -> dict:
        return {
            "schema": "skillrecall/collection/v1",
            "source": self.source,
            "skills": [a.as_dict(detail) for a in sorted(self.skills, key=lambda a: a.recall.value)],
            "sibling_pairs": [p.as_dict() for p in self.sibling_pairs],
            "failures": [{"skill": n, "error": e} for n, e in self.failures],
            "elapsed_seconds": round(self.elapsed, 3),
        }


def assess_collection(
    base: Options, skill_dirs: list[Path], source: str, catalog_source: str = "", workers: int = 4, progress=None
) -> Collection:
    """Assess every skill in a collection, each competing against its siblings."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    t0 = time.perf_counter()
    dirs = [str(d) for d in skill_dirs]

    def one(d: str) -> Assessment:
        opts = Options(**{f.name: getattr(base, f.name) for f in fields(base)})
        opts.progress = None  # the collection ticker owns the display
        opts.skill_path = d
        opts.sibling_dirs = [x for x in dirs if x != d]
        opts.corpus_dirs = list(base.corpus_dirs)
        if catalog_source:
            opts.catalog_id = f"{catalog_source}/{Path(d).name}"
            opts.source_url = f"https://github.com/{catalog_source}"
        return assess(opts)

    results: list[Assessment] = []
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(one, d): d for d in dirs}
        for done, fut in enumerate(as_completed(futures), start=1):
            d = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:  # one bad skill must not sink the collection
                failures.append((Path(d).name, str(e)))
            if progress:
                progress(done, len(dirs), Path(d).name)
    results.sort(key=lambda a: a.skill.name)
    return Collection(source, results, failures, time.perf_counter() - t0)
