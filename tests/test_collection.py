from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from skillrecall.assess import Options, assess_collection
from skillrecall.remote import local_shape, skill_dirs_in

FIXTURES = Path(__file__).parent / "fixtures"


def test_local_shape_detects_skill_and_collection():
    shape, dirs = local_shape(str(FIXTURES / "landing-clarity"))
    assert shape == "skill" and dirs[0].name == "landing-clarity"
    shape, dirs = local_shape(str(FIXTURES / "landing-clarity" / "SKILL.md"))
    assert shape == "skill"
    shape, dirs = local_shape(str(FIXTURES))
    assert shape == "collection" and len(dirs) == 5


def test_skill_dirs_in_tree_skips_hidden_and_vendored():
    tree = [
        "SKILL.md",
        "skills/a/SKILL.md",
        "skills/b/SKILL.md",
        ".git/x/SKILL.md",
        "node_modules/p/SKILL.md",
        "skills/a/references/notes.md",
    ]
    assert skill_dirs_in(tree) == ["", "skills/a", "skills/b"]
    assert skill_dirs_in(tree, "skills") == ["skills/a", "skills/b"]


def test_collection_assesses_every_skill_against_siblings(tmp_path):
    base = Options(skill_path="", catalog=False, state_root=tmp_path, tasks=40, composition=20)
    dirs = sorted(p for p in FIXTURES.iterdir() if (p / "SKILL.md").is_file())
    coll = assess_collection(base, dirs, str(FIXTURES), workers=2)
    assert len(coll.skills) == 5 and not coll.failures
    for a in coll.skills:
        assert len(a.neighbours) == 4
        assert all(n.origin == "sibling" for n in a.neighbours)
    d = coll.as_dict()
    json.dumps(d)
    assert d["schema"] == "skillrecall/collection/v1"
    assert [s["skill"]["name"] for s in d["skills"]] == [a.skill.name for a in sorted(coll.skills, key=lambda a: a.recall.value)]


def test_cli_collection_json():
    cmd = [sys.executable, "-m", "skillrecall", "assess", str(FIXTURES), "--no-catalog", "--no-save", "--format", "json", "--tasks", "30"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    assert d["schema"] == "skillrecall/collection/v1"
    assert len(d["skills"]) == 5
