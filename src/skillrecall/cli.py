"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skillrecall",
        description="Measure how reliably a skill gets picked next to other skills, and what to change.",
        epilog=(
            "A skill reference can be a local directory, a SKILL.md path, a skills.sh link, a GitHub link, or\n"
            "owner/repo/skill. A reference that holds several skills (a repository, a skills/ directory) is\n"
            "assessed as a collection: every skill competes against its siblings and you get one summary table.\n\n"
            "examples:\n"
            "  skillrecall assess ./my-skill --installed\n"
            "  skillrecall assess https://skills.sh/owner/repo/skill\n"
            "  skillrecall assess https://github.com/owner/repo/skill\n"
            "  skillrecall assess owner/repo                # whole repo, one row per skill\n"
            "  skillrecall assess ./skills --no-catalog     # local collection, siblings only\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"skillrecall {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("assess", help="assess a skill, or every skill in a repository or directory")
    a.add_argument(
        "skill", help="skill directory, SKILL.md path, skills.sh link, GitHub link, owner/repo/skill, or a repository/directory of skills"
    )
    a.add_argument("--workers", type=int, default=4, help="skills assessed in parallel for a collection (default 4)")
    a.add_argument("--quiet", action="store_true", help="no progress ticker on stderr")
    a.add_argument("--corpus", action="append", default=[], metavar="DIR", help="directory of skills to compete against (repeatable)")
    a.add_argument("--installed", action="store_true", help="also compete against skills installed on this machine")
    a.add_argument("--no-catalog", action="store_true", help="skip the public catalog")
    a.add_argument("--offline", action="store_true", help="use only cached catalog data; never touch the network")
    a.add_argument("--neighbours", type=int, default=40, help="how many catalog neighbours to fetch (default 40)")
    a.add_argument("--tasks", type=int, default=100, help="how many of your own tasks to sample (default 100)")
    a.add_argument("--tasks-file", metavar="FILE", help="real requests, one per line, used before generated ones")
    a.add_argument("--format", choices=("human", "json"), default="human")
    a.add_argument("--detail", choices=("simple", "detailed"), default="simple")
    a.add_argument("--explain", action="store_true", help="append how the numbers were measured (human format)")
    a.add_argument("--dense", action="store_true", help="add the local embedding scorer (needs the 'dense' extra)")
    a.add_argument("--dense-model", default=None)
    a.add_argument("--reference", action="store_true", help="also ask a model to route a sample of tasks (needs the 'router' extra)")
    a.add_argument("--reference-model", default="claude-opus-5")
    a.add_argument("--reference-n", type=int, default=30)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--no-save", action="store_true", help="do not record this run in the history")
    a.add_argument("--state-dir", default=None, help="where run history is kept")
    a.add_argument("--timeout", type=float, default=12.0, help="network timeout in seconds")
    a.add_argument("-o", "--output", metavar="FILE", help="write the report here instead of stdout")

    h = sub.add_parser("history", help="show previous runs for a skill")
    h.add_argument("skill", help="the same reference you passed to assess")
    h.add_argument("--state-dir", default=None)
    h.add_argument("--format", choices=("human", "json"), default="human")
    return p


def _assess(ns: argparse.Namespace) -> int:
    from .assess import Options, assess, assess_collection
    from .progress import Ticker
    from .remote import is_remote, local_shape, resolve
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
        if is_remote(ns.skill):
            shape, dirs, ref = resolve(ns.skill, timeout=ns.timeout)
            source = ref.url
            catalog_source = ref.source
            if shape == "skill":
                opts.skill_path = str(dirs[0])
                opts.source_url = ref.url
                opts.catalog_id = ref.catalog_id
        else:
            shape, dirs = local_shape(ns.skill)
            source = str(dirs[0].parent if shape == "collection" else dirs[0])
            if shape == "skill":
                opts.skill_path = str(dirs[0])
        if shape == "collection":
            ticker.progress(0, len(dirs), "starting")
            coll = assess_collection(
                opts,
                dirs,
                source,
                catalog_source,
                workers=ns.workers,
                progress=lambda done, total, name: ticker.progress(done, total, f"finished {name}"),
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
