from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from skillrecall.assess import Options, assess
from skillrecall.edits import candidate_edits
from skillrecall.scoring import Doc, Router, build_doc, evaluate, parse_guards
from skillrecall.skill import load_skill, parse_frontmatter
from skillrecall.stats import paired_delta, rate
from skillrecall.tasks import Task, sample_tasks, seeds_for
from skillrecall.text import TokenCounter, name_similarity, sentences, terms, tokens

FIXTURES = Path(__file__).parent / "fixtures"


# --- text -----------------------------------------------------------------


def test_tokens_drop_function_words_and_short_tokens():
    assert tokens("The landing page is for a visitor") == ["landing", "page", "visitor"]


def test_terms_include_bigrams():
    assert "landing_page" in terms("landing page audit")


def test_sentences_split_on_terminators():
    assert len(sentences("First one. Second one? Third!")) == 3


def test_token_estimate_is_monotone_and_reasonable():
    c = TokenCounter(prefer_exact=False)
    short = c.count("Audit a landing page.")
    long = c.count("Audit a landing page for clarity, jargon, and curse of knowledge failures.")
    assert 3 <= short < long <= 30


def test_name_similarity_bounds():
    assert name_similarity("landing-page", "landing-page") == 1.0
    assert name_similarity("landing-page", "sql-migrations") < 0.2


# --- skill parsing ----------------------------------------------------------


def test_frontmatter_handles_folded_and_quoted_values():
    raw = '---\nname: "demo"\ndescription: >\n  First line\n  second line.\nmetadata:\n  version: "1.0"\n---\n# Body\n'
    fm, body = parse_frontmatter(raw)
    assert fm["name"] == "demo"
    assert fm["description"] == "First line second line."
    assert fm["metadata.version"] == "1.0"
    assert body.startswith("# Body")


def test_load_skill_measures_files():
    sk = load_skill(FIXTURES / "landing-clarity")
    assert sk.name == "landing-clarity"
    assert sk.resident_tokens > sk.description_tokens > 0
    assert [r.path for r in sk.reference_files] == ["references/rubric.md"]
    assert any(s.heading == "When to use" for s in sk.sections)


# --- tasks ------------------------------------------------------------------


def test_seeds_come_from_usage_sections_not_procedure():
    sk = load_skill(FIXTURES / "landing-clarity")
    seeds, weak = seeds_for(sk.body, sk.description)
    assert not weak
    assert "Review my landing page for clarity" in seeds
    assert not any(s.startswith("Read the page as a stranger") for s in seeds)


def test_sampling_is_deterministic():
    seeds = ["review my landing page", "smell test the hero copy"]
    a = [t.text for t in sample_tasks(seeds, 20, 0, "body", seed=3)]
    b = [t.text for t in sample_tasks(seeds, 20, 0, "body", seed=3)]
    assert a == b and len(a) == 20


# --- scoring ----------------------------------------------------------------


def _docs():
    known = {"landing-clarity": 0, "pricing-page-audit": 1, "sql-migrations": 2}
    return [
        build_doc("a", "landing-clarity", "Audit a landing page for clarity and jargon.", known),
        build_doc("b", "pricing-page-audit", "Assess a pricing page against anchoring and decoy best practice.", known),
        build_doc("c", "sql-migrations", "Write safe database schema migrations with rollback scripts.", known),
    ]


def test_router_picks_the_matching_skill():
    r = Router(_docs())
    s = r.score("review my landing page for jargon")
    assert r.winner(s) == 0
    s = r.score("write a rollback script for this migration")
    assert r.winner(s) == 2


def test_guard_yields_to_named_skill():
    known = {"landing-clarity": 0, "pricing-page-audit": 1}
    guards, stripped = parse_guards("Audit a landing page for clarity. Not for pricing pages; use pricing-page-audit for that.", known)
    assert len(guards) == 1 and guards[0].target == "pricing-page-audit"
    assert "pricing-page-audit" not in stripped
    docs = [
        build_doc("a", "landing-clarity", "Audit a landing page for clarity. Not for pricing pages; use pricing-page-audit for that.", known),
        build_doc("b", "pricing-page-audit", "Assess a pricing page layout.", known),
    ]
    r = Router(docs)
    assert r.winner(r.score("audit our pricing page")) == 1


def test_evaluate_counts_hits_and_ranks():
    r = Router(_docs())
    own = [Task("audit the landing page for jargon", 0, "body"), Task("is this page clear", 0, "body")]
    adv = [Task("check our pricing page anchoring", 1, "neighbour")]
    out = evaluate(r, 0, own, adv, [])
    assert out.own_hits[0] is True
    assert out.adv_hits == [False]
    assert out.own_ranks[0] == 1


# --- stats ------------------------------------------------------------------


def test_rate_interval_contains_point_estimate():
    r = rate([True] * 70 + [False] * 30)
    assert r.low <= r.value <= r.high
    assert 0.6 < r.value < 0.8


def test_paired_delta_detects_change():
    before = [False] * 50 + [True] * 50
    after = [True] * 100
    d = paired_delta(before, after)
    assert d.significant and d.delta == pytest.approx(0.5)
    same = paired_delta(before, before)
    assert not same.significant and same.delta == 0.0


# --- edits ------------------------------------------------------------------


def test_candidate_edits_cover_each_sentence_and_truncations():
    sk = load_skill(FIXTURES / "landing-clarity")
    edits = candidate_edits(sk, [], ["headline"], ["clarity"], ["landing-page"])
    kinds = {e.kind for e in edits}
    assert {"remove_sentence", "truncate", "add_terms"} <= kinds
    n_sent = len(sentences(sk.description))
    assert sum(1 for e in edits if e.kind == "remove_sentence") == n_sent


# --- end to end -------------------------------------------------------------


def test_assess_offline_against_fixture_corpus(tmp_path):
    opts = Options(skill_path=str(FIXTURES / "landing-clarity"), corpus_dirs=[str(FIXTURES)], catalog=False, state_root=tmp_path, tasks=60)
    a = assess(opts)
    assert len(a.neighbours) == 4
    assert a.recall.value >= 0.8
    assert a.false_positives.value <= 0.1
    assert a.recall.n == 60
    d = a.as_dict(True)
    json.dumps(d)
    assert d["schema"] == "skillrecall/v1"
    # A second run records movement against the first.
    b = assess(opts)
    assert b.previous_run == a.generated_at
    assert b.movement


def test_suggested_description_never_measures_worse(tmp_path):
    opts = Options(skill_path=str(FIXTURES / "landing-clarity"), corpus_dirs=[str(FIXTURES)], catalog=False, save_snapshot=False, tasks=60)
    a = assess(opts)
    assert a.suggested_recall_delta >= -0.03
    for e in a.edits:
        assert e.accepted and e.verdict != "would hurt"


def test_cli_json_output(tmp_path):
    cmd = [sys.executable, "-m", "skillrecall", "assess", str(FIXTURES / "sql-migrations"), "--corpus", str(FIXTURES), "--no-catalog", "--no-save", "--format", "json", "--tasks", "40"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    assert d["skill"]["name"] == "sql-migrations"
    assert d["pickup"]["recall"]["n"] == 40
