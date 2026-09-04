"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skillrecall", description="Measure how reliably a skill gets picked next to other skills, and what to change.")
    p.add_argument("--version", action="version", version=f"skillrecall {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("assess", help="assess one skill directory")
    a.add_argument("skill", help="path to the skill directory or its SKILL.md")
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
    h.add_argument("skill")
    h.add_argument("--state-dir", default=None)
    h.add_argument("--format", choices=("human", "json"), default="human")
    return p


def _assess(ns: argparse.Namespace) -> int:
    from .assess import Options, assess
    from .report import render_human, render_json

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
        result = assess(opts)
    except FileNotFoundError as e:
        print(f"skillrecall: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"skillrecall: {e}", file=sys.stderr)
        return 2
    text = render_json(result, ns.detail) if ns.format == "json" else render_human(result, ns.detail, ns.explain)
    if ns.output:
        Path(ns.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def _history(ns: argparse.Namespace) -> int:
    import json

    from . import snapshots

    root = Path(ns.state_dir) if ns.state_dir else None
    runs = snapshots.history(ns.skill, root)
    if ns.format == "json":
        print(json.dumps(runs, indent=2))
        return 0
    if not runs:
        print("No previous runs.")
        return 0
    print(f"{'when':<22} {'picked':>7} {'mistaken':>9} {'tokens':>7} {'lines':>6}  name")
    for r in runs:
        print(f"{r.get('generated_at', ''):<22} {r.get('recall', 0):>7.2f} {r.get('false_positives', 0):>9.2f} {r.get('resident_tokens', 0):>7} {r.get('body_lines', 0):>6}  {r.get('name', '')}")
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
