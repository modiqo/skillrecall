"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__

COLLECTION_LIMIT = 8  # above this, a collection asks you to choose unless --all


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skillrecall",
        description=(
            "Measure how reliably a skill gets picked when it is installed next to other skills,\n"
            "and get a short, ranked list of edits that make it get picked more."
        ),
        epilog=(
            "what a skill reference can be:\n"
            "  ./my-skill                                a local directory, or the path to its SKILL.md\n"
            "  https://skills.sh/owner/repo/skill        a catalog page\n"
            "  https://github.com/owner/repo/skill       a GitHub link; tree/ and blob/ links work too\n"
            "  owner/repo/skill                          shorthand for either of the above\n"
            "  owner/repo, ./skills                      a collection: every skill in it\n\n"
            "collections of up to 8 skills run at once, each competing against its siblings and the\n"
            "catalog. Larger ones list their skills and ask you to choose (--pick a,b) or insist (--all).\n\n"
            "examples:\n"
            "  skillrecall assess ./my-skill --installed\n"
            "  skillrecall assess https://skills.sh/owner/repo/skill\n"
            "  skillrecall assess owner/repo --pick code-review,triage\n"
            "  skillrecall assess ./skills --no-catalog --format json -o report.json\n"
            "  skillrecall history ./my-skill\n\n"
            "run `skillrecall assess --help` for every option."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"skillrecall {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    a = sub.add_parser(
        "assess",
        help="assess one skill, or the skills in a repository or directory",
        description="Assess a skill against its competitors and report what to change.",
        epilog=(
            "examples:\n"
            "  skillrecall assess ./my-skill                       against the public catalog\n"
            "  skillrecall assess ./my-skill --installed           plus everything installed here\n"
            "  skillrecall assess ./my-skill --corpus ../team      plus a directory of your own skills\n"
            "  skillrecall assess ./my-skill --tasks-file asks.txt with real requests, one per line\n"
            "  skillrecall assess owner/repo/skill --detail detailed --explain\n"
            "  skillrecall assess owner/repo --pick a,b            a few skills, competing with each other\n"
            "  skillrecall assess owner/repo --all                 every skill; several minutes\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    a.add_argument(
        "skill",
        metavar="SKILL",
        help="what to assess: a directory, a SKILL.md path, a skills.sh or GitHub link, owner/repo/skill, or a repository or directory of skills",
    )

    g = a.add_argument_group("collections", "when SKILL holds several skills")
    g.add_argument("--pick", metavar="A,B", help="assess only these skills, by name; they also compete against each other")
    g.add_argument(
        "--all",
        action="store_true",
        help=f"assess every skill even when there are more than {COLLECTION_LIMIT} (default: stop and list them)",
    )
    g.add_argument("--workers", metavar="N", type=int, default=4, help="skills assessed in parallel (default 4)")

    g = a.add_argument_group("competition", "who the skill is measured against")
    g.add_argument(
        "--installed",
        action="store_true",
        help="add every skill installed on this machine (~/.claude/skills, ~/.agents/skills, and others)",
    )
    g.add_argument("--corpus", action="append", default=[], metavar="DIR", help="add every skill in this directory; repeatable")
    g.add_argument("--neighbours", metavar="N", type=int, default=40, help="how many catalog competitors to fetch (default 40)")
    g.add_argument("--no-catalog", action="store_true", help="do not search the public catalog; use only --installed and --corpus")
    g.add_argument("--offline", action="store_true", help="use cached catalog data only; never touch the network")

    g = a.add_argument_group("requests", "the sample of requests the skill is scored on")
    g.add_argument(
        "--tasks",
        metavar="N",
        type=int,
        default=100,
        help="how many of the skill's own requests to sample; more gives a tighter confidence range (default 100)",
    )
    g.add_argument("--tasks-file", metavar="FILE", help="real requests, one per line; used before any generated ones")
    g.add_argument(
        "--seed", metavar="N", type=int, default=7, help="sampling seed; same seed and same competitors give the same numbers (default 7)"
    )

    g = a.add_argument_group("output")
    g.add_argument("--format", choices=("human", "json"), default="human", help="human-readable report or JSON (default human)")
    g.add_argument(
        "--detail",
        choices=("simple", "detailed"),
        default="simple",
        help="simple shows the advice; detailed adds sizes, every competitor, every edit tried, and sample requests (default simple)",
    )
    g.add_argument(
        "--explain", action="store_true", help="append how the numbers were measured, with raw rates and intervals (human format)"
    )
    g.add_argument("-o", "--output", metavar="FILE", help="write the report to this file instead of stdout")
    g.add_argument("--quiet", action="store_true", help="no progress ticker on stderr (it is already silent when stderr is not a terminal)")

    g = a.add_argument_group("optional scorers", "each needs an extra: pip install 'skillrecall[dense]' or '[router]'")
    g.add_argument("--dense", action="store_true", help="add a local embedding model as a second scorer, for meaning beyond shared words")
    g.add_argument(
        "--dense-model", metavar="NAME", default=None, help="embedding model to use with --dense (default BAAI/bge-small-en-v1.5)"
    )
    g.add_argument(
        "--reference",
        action="store_true",
        help="also ask a model to route a sample of requests, and report its pickup rate and agreement with the local scorer",
    )
    g.add_argument("--reference-model", metavar="ID", default="claude-opus-5", help="model for --reference (default claude-opus-5)")
    g.add_argument("--reference-n", metavar="N", type=int, default=30, help="requests to send to the reference model (default 30)")

    g = a.add_argument_group("history and network")
    g.add_argument("--no-save", action="store_true", help="do not record this run, so the next run will not report movement against it")
    g.add_argument("--state-dir", metavar="DIR", default=None, help="where run history is kept (default ~/.local/state/skillrecall)")
    g.add_argument("--timeout", metavar="SECONDS", type=float, default=12.0, help="per-request network timeout (default 12)")

    h = sub.add_parser(
        "history",
        help="show previous runs of a skill",
        description="Show every recorded run of a skill: when, pickup rate, mistaken pickups, size.",
    )
    h.add_argument("skill", metavar="SKILL", help="the same reference you passed to assess")
    h.add_argument("--state-dir", metavar="DIR", default=None, help="where run history is kept")
    h.add_argument("--format", choices=("human", "json"), default="human", help="table or JSON (default human)")
    return p


def _assess(ns: argparse.Namespace) -> int:
    from .assess import Options, assess, assess_collection, display_name
    from .progress import Ticker
    from .remote import guidance, is_remote, local_shape, resolve
    from .report import render_collection_human, render_collection_json, render_human, render_json

    ticker = Ticker(enabled=False if ns.quiet else None)

    opts = Options(
        skill_path=ns.skill,
        corpus_dirs=ns.corpus,
        include_installed=ns.installed,
        catalog=not ns.no_catalog,
        offline=ns.offline,
        neighbours=ns.neighbours,
        tasks=ns.tasks,
        tasks_file=ns.tasks_file,
        seed=ns.seed,
        dense=ns.dense,
        dense_model=ns.dense_model,
        reference=ns.reference,
        reference_model=ns.reference_model,
        reference_n=ns.reference_n,
        save_snapshot=not ns.no_save,
        state_root=Path(ns.state_dir) if ns.state_dir else None,
        timeout=ns.timeout,
    )
    try:
        ticker.start("resolving the reference")
        # Detect the shape first: one skill, or a collection of them.
        catalog_source = ""
        pick = [p for p in (ns.pick or "").split(",") if p.strip()]
        if is_remote(ns.skill):
            shape, found, ref = resolve(ns.skill, timeout=ns.timeout)
            source = ref.url
            catalog_source = ref.source
            if shape == "skill":
                dirs = [found]
                opts.skill_path = str(found)
                opts.source_url = ref.url
                opts.catalog_id = ref.catalog_id
            else:
                names = found.names
                if not pick and not ns.all and len(names) > COLLECTION_LIMIT:
                    ticker.stop()
                    return _fail(guidance(ref.source, names, too_many=True))
                ticker.stage(f"fetching {len(pick) or len(names)} skills")
                dirs = found.fetch(pick or None, timeout=ns.timeout)
        else:
            shape, dirs = local_shape(ns.skill)
            source = str(dirs[0].parent if shape == "collection" else dirs[0])
            if shape == "skill":
                opts.skill_path = str(dirs[0])
            else:
                names = [display_name(d) for d in dirs]
                if pick:
                    wanted = {p.strip() for p in pick}
                    dirs = [d for d in dirs if display_name(d) in wanted]
                    missing = sorted(wanted - {display_name(d) for d in dirs})
                    if missing:
                        ticker.stop()
                        return _fail(guidance(source, names, missing=missing[0], local=True))
                elif not ns.all and len(names) > COLLECTION_LIMIT:
                    ticker.stop()
                    return _fail(guidance(source, names, too_many=True, local=True))
        if shape == "collection" and len(dirs) == 1:
            shape = "skill"
            opts.skill_path = str(dirs[0])
        if shape == "collection":
            ticker.progress(0, len(dirs), "starting")
            coll = assess_collection(
                opts,
                dirs,
                source,
                catalog_source,
                workers=ns.workers,
                progress=ticker.finish,
                stage=ticker.update,
            )
            text = render_collection_json(coll, ns.detail) if ns.format == "json" else render_collection_human(coll, ns.detail, ns.explain)
        else:
            opts.progress = ticker.stage
            result = assess(opts)
            text = render_json(result, ns.detail) if ns.format == "json" else render_human(result, ns.detail, ns.explain)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        ticker.stop()
        return _fail(str(e))
    finally:
        ticker.stop()
    if ns.output:
        Path(ns.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def _fail(message: str) -> int:
    lines = message.splitlines() or [message]
    print(f"skillrecall: {lines[0]}", file=sys.stderr)
    for line in lines[1:]:
        print(f"  {line}" if line else "", file=sys.stderr)
    return 2


def _history(ns: argparse.Namespace) -> int:
    import json

    from . import snapshots
    from .remote import is_remote, materialise

    root = Path(ns.state_dir) if ns.state_dir else None
    key = ns.skill
    if is_remote(key):
        try:
            key = str(materialise(key)[0])
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            return _fail(str(e))
    runs = snapshots.history(key, root)
    if ns.format == "json":
        print(json.dumps(runs, indent=2))
        return 0
    if not runs:
        print("No previous runs.")
        return 0
    print(f"{'when':<22} {'picked':>7} {'mistaken':>9} {'tokens':>7} {'lines':>6}  name")
    for r in runs:
        print(
            f"{r.get('generated_at', ''):<22} {r.get('recall', 0):>7.2f} {r.get('false_positives', 0):>9.2f} {r.get('resident_tokens', 0):>7} {r.get('body_lines', 0):>6}  {r.get('name', '')}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    if ns.command == "assess":
        return _assess(ns)
    if ns.command == "history":
        return _history(ns)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
