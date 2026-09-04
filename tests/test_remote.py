from __future__ import annotations

import pytest

from skillrecall.remote import RemoteRef, _locate, is_remote, parse_ref


def test_local_paths_are_not_remote(tmp_path):
    assert not is_remote(str(tmp_path))
    assert not is_remote("./skills/foo")
    assert not is_remote("~/skills/foo")


def test_remote_forms_are_detected():
    for ref in (
        "https://skills.sh/owner/repo/skill",
        "https://www.skills.sh/owner/repo/skill",
        "skills.sh/owner/repo/skill",
        "https://github.com/owner/repo",
        "https://github.com/owner/repo/tree/main/skills/skill",
        "github.com/owner/repo/blob/main/skills/skill/SKILL.md",
        "owner/repo/skill",
        "owner/repo",
    ):
        assert is_remote(ref), ref


def test_parse_skills_sh_link():
    r = parse_ref("https://skills.sh/owner/repo/skill")
    assert r.source == "owner/repo" and r.catalog_id == "owner/repo/skill" and r.ref == "HEAD" and r.path == ""


def test_parse_github_tree_and_blob():
    r = parse_ref("https://github.com/owner/repo/tree/dev/skills/skill")
    assert (r.source, r.ref, r.path) == ("owner/repo", "dev", "skills/skill")
    b = parse_ref("https://github.com/owner/repo/blob/dev/skills/skill/SKILL.md")
    assert (b.source, b.ref, b.path) == ("owner/repo", "dev", "skills/skill")


def test_parse_shorthand():
    r = parse_ref("owner/repo/skill")
    assert r.catalog_id == "owner/repo/skill"
    root = parse_ref("owner/repo")
    assert root.catalog_id == "" and root.path == ""
    deep = parse_ref("owner/repo/skills/skill")
    assert deep.path == "skills/skill" and deep.catalog_id == ""


def test_parse_rejects_unknown_host():
    with pytest.raises(ValueError):
        parse_ref("https://example.com/owner/repo/skill")


def test_locate_prefers_common_paths_then_suffix():
    r = RemoteRef("o/r", "", "HEAD", "", "o/r/alpha")
    tree = ["README.md", "packs/x/alpha/SKILL.md", "skills/alpha/SKILL.md"]
    assert _locate(r, tree) == "skills/alpha"
    tree2 = ["README.md", "packs/x/alpha/SKILL.md"]
    assert _locate(r, tree2) == "packs/x/alpha"


def test_locate_root_and_ambiguity():
    root = RemoteRef("o/r", "", "HEAD", "", "")
    assert _locate(root, ["SKILL.md", "notes.md"]) == ""
    assert _locate(root, ["skills/only/SKILL.md"]) == "skills/only"
    with pytest.raises(ValueError):
        _locate(root, ["skills/a/SKILL.md", "skills/b/SKILL.md"])
    with pytest.raises(FileNotFoundError):
        _locate(RemoteRef("o/r", "nope", "HEAD", "", ""), ["skills/a/SKILL.md"])
