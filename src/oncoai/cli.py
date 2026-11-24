"""Command line: fetch | qc | profile | cohort | train | predict | demo | serve."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .model import RESEARCH_USE


def _studies(value: str) -> list[str]:
    from .data import BENCHMARK, load_manifest

    if value == "benchmark":
        return list(BENCHMARK)
    if value == "all":
        return sorted(load_manifest().studies)
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def cmd_fetch(args: argparse.Namespace) -> int:
    from .data import fetch, load_manifest

    manifest = load_manifest()
    codes = _studies(args.studies)
    print(f"Source: {manifest.repository} @ {manifest.commit[:12]} -> {args.dest}")
    results = fetch(codes, Path(args.dest), manifest)
    failed = 0
    for key, status in results.items():
        print(f"  {key:<22} {status}")
        failed += status.startswith("failed")
    print(f"{len(results) - failed} ok, {failed} failed. Data licences: see {manifest.repository} (ODbL 1.0) and TCGA terms.")
    return 1 if failed else 0


def cmd_qc(args: argparse.Namespace) -> int:
    from .maf import Filters, MafError, read_maf

    try:
        data = read_maf(Path(args.file), Filters(min_vaf=args.min_vaf, min_alt_reads=args.min_alt_reads))
    except (MafError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = data.qc.as_dict()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"rows {report['rows']}  kept {report['kept']}  samples {report['samples']}")
        for reason, n in report["rejected"].items():
            print(f"  rejected {n:>6}  {reason} (first at line {report['first_line_per_reason'][reason]})")
        for w in report["warnings"]:
            print(f"  warning  {w}")
    return 0 if report["kept"] else 1


def cmd_profile(args: argparse.Namespace) -> int:
    from .features import profiles
    from .maf import read_maf

    data = read_maf(Path(args.file))
    out = [p.summary() for p in profiles(data.by_sample(), args.exome_mb).values()]
    print(json.dumps(out, indent=2))
    return 0


def cmd_cohort(args: argparse.Namespace) -> int:
    from .cohort import pair_tests, recurrence
    from .data import load_study

    cohort = load_study(Path(args.data), args.study.lower())
    print(f"{args.study.upper()}: {len(cohort)} samples")
    for g in recurrence(cohort.profiles, args.top):
        changes = ", ".join(f"{c} ({n})" for c, n in g.top_changes)
        print(f"  {g.gene:<10} {g.fraction:6.1%}  {changes}")
    if args.pairs:
        print("Pairs (Fisher exact, BH-adjusted; screening only, see docs):")
        for t in pair_tests(cohort.profiles, top=args.top, exclude_hypermutated_above=args.max_tmb)[: args.pairs]:
            d = t.as_dict()
            print(f"  {t.gene_a:<8} {t.gene_b:<8} {d['tendency']:<19} OR {d['odds_ratio']}  q {d['q']}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .data import load_cohort, load_manifest
    from .model import TrainParams
    from .training import train

    codes = _studies(args.studies)
    cohort = load_cohort(Path(args.data), codes)
    manifest = load_manifest()
    params = TrainParams(use_sbs96=args.sbs96)
    model, report = train(
        cohort.profiles,
        cohort.labels,
        cohort.patients,
        params,
        n_splits=args.folds,
        seed=args.seed,
        target_accuracy=args.target_accuracy,
        with_boosting=args.boosting,
        data_source={"repository": manifest.repository, "commit": manifest.commit, "studies": codes},
    )
    model.save(Path(args.out))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    m = report["cross_validation"]["methods"]["logistic"]["pooled"]
    a = report["abstention"]
    print(f"{len(cohort)} samples, {len(set(cohort.labels))} labels")
    print(f"cross-validated: accuracy {m['accuracy']:.3f}  balanced {m['balanced_accuracy']:.3f}", end="")
    print(f"  macro-F1 {m['macro_f1']:.3f}  ECE {m['ece']:.3f}")
    coverage, correct = a["out_of_fold_coverage"], a["out_of_fold_accuracy_when_answered"]
    print(f"abstain below {a['threshold']:.2f}: answers {coverage:.0%} of samples, {correct:.1%} correct")
    print(f"model -> {args.out}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .features import profiles
    from .maf import read_maf
    from .model import Model

    model = Model.load(Path(args.model))
    data = read_maf(Path(args.file))
    preds = model.predict(list(profiles(data.by_sample()).values()))
    print(json.dumps({"notice": RESEARCH_USE, "qc": data.qc.as_dict(), "predictions": [p.as_dict() for p in preds]}, indent=2))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run

    return run(Path(args.out) if args.out else None)


def cmd_serve(args: argparse.Namespace) -> int:
    import logging

    import uvicorn

    from .api import KeyStore, create_app, demo_model
    from .model import Model

    try:
        keys = KeyStore(os.environ.get("ONCOAI_API_KEYS", ""))
    except ValueError as exc:
        print(f"ONCOAI_API_KEYS: {exc}. Example: viewer:<24+ chars>,analyst:<...>,admin:<...>", file=sys.stderr)
        return 2
    model_path = args.model or os.environ.get("ONCOAI_MODEL")
    example_path = os.environ.get("ONCOAI_EXAMPLE_MAF")
    if model_path:
        model, example = Model.load(Path(model_path)), None
        print(f"model: {model_path} ({len(model.classes)} labels)")
    else:
        model, example = demo_model()
        print("model: built-in synthetic demo model (set ONCOAI_MODEL to serve a trained model)")
    if example_path:
        example = Path(example_path).read_text(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = create_app(model, keys, Path(model_path) if model_path else None, example)
    host = args.host or os.environ.get("ONCOAI_HOST", "127.0.0.1")
    port = args.port or int(os.environ.get("ONCOAI_PORT", "8080"))
    print(f"{RESEARCH_USE}\nconsole: http://{host}:{port}/dashboard")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oncoai", description=f"Precision oncology research platform. {RESEARCH_USE}")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download public study files and verify their checksums")
    p.add_argument("--studies", default="benchmark", help="'benchmark', 'all' or comma-separated codes")
    p.add_argument("--dest", default="data/pancan")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("qc", help="check a MAF file")
    p.add_argument("file")
    p.add_argument("--min-vaf", type=float, default=0.0)
    p.add_argument("--min-alt-reads", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_qc)

    p = sub.add_parser("profile", help="per-sample TMB and substitution profile")
    p.add_argument("file")
    p.add_argument("--exome-mb", type=float, default=30.0)
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("cohort", help="recurrent genes and gene pairs in one fetched study")
    p.add_argument("study")
    p.add_argument("--data", default="data/pancan")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--pairs", type=int, default=0, help="show this many gene-pair tests")
    p.add_argument("--max-tmb", type=float, default=None, help="leave out samples above this TMB for pair tests")
    p.set_defaults(func=cmd_cohort)

    p = sub.add_parser("train", help="cross-validate, then fit the served model on all samples")
    p.add_argument("--data", default="data/pancan")
    p.add_argument("--studies", default="benchmark")
    p.add_argument("--out", default="model.json")
    p.add_argument("--report", default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--target-accuracy", type=float, default=0.9)
    p.add_argument("--boosting", action="store_true", help="also cross-validate gradient boosting for comparison")
    p.add_argument("--sbs96", action="store_true", help="add the 96 trinucleotide channels (worse on the benchmark)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="tumour-type probabilities for each sample in a MAF")
    p.add_argument("file")
    p.add_argument("--model", default="model.json")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("demo", help="offline end-to-end run on synthetic data")
    p.add_argument("--out", default=None, help="write the demo model here")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("serve", help="run the API and console (keys and model from ONCOAI_* environment variables)")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--model", default=None, help="model JSON; default ONCOAI_MODEL, else a synthetic demo model")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
