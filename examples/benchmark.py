"""Reproduce the numbers in docs/data/validation.md.

    oncoai fetch --studies benchmark --dest data/pancan
    python examples/benchmark.py --data data/pancan

Steps: (1) TMB agreement with the value cBioPortal publishes per sample, (2) grouped cross-validation
over three seeds with baselines, (3) ablation with the 96 trinucleotide channels, (4) gene-pair
checks in two cohorts against patterns reported by the TCGA working groups.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
from pathlib import Path

import numpy as np

from oncoai.cohort import pair_tests
from oncoai.data import BENCHMARK, load_cohort, load_study
from oncoai.evaluate import cross_validate
from oncoai.model import TrainParams

KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy", "log_loss", "ece")


def tmb_agreement(cohort) -> None:
    diffs = np.array([round((p.tmb - cohort.published_tmb[p.sample]) * 30) for p in cohort.profiles if p.sample in cohort.published_tmb])
    ours = np.array([p.tmb for p in cohort.profiles if p.sample in cohort.published_tmb])
    theirs = np.array([cohort.published_tmb[p.sample] for p in cohort.profiles if p.sample in cohort.published_tmb])
    r = np.corrcoef(np.log1p(ours), np.log1p(theirs))[0, 1]
    print("1. TMB vs cBioPortal TMB_NONSYNONYMOUS")
    print(f"   samples {len(diffs)}; identical {np.mean(diffs == 0):.1%}; within 2 mutations {np.mean(np.abs(diffs) <= 2):.1%};")
    print(f"   largest difference {int(np.abs(diffs).max())} mutations; Pearson r on log(1+TMB) {r:.4f}")


def cv_over_seeds(cohort, seeds=(7, 11, 23)) -> None:
    print("\n2. Grouped 5-fold cross-validation, C chosen in inner folds, three seeds")
    runs: dict[str, list[dict]] = {}
    for seed in seeds:
        res = cross_validate(cohort.profiles, cohort.labels, cohort.patients, TrainParams(), seed=seed, with_boosting=False)
        for name, s in res.summary().items():
            runs.setdefault(name, []).append(s["pooled"])
    for name, items in runs.items():
        cells = []
        for k in KEYS:
            vals = [m[k] for m in items]
            cells.append(f"{k} {statistics.mean(vals):.3f}±{statistics.pstdev(vals):.3f}")
        print(f"   {name:<24} " + "  ".join(cells))


def sbs96_ablation(cohort) -> None:
    print("\n3. Ablation: add the 96 trinucleotide channels (seed 7, C = 1.0)")
    for label, params in (("without SBS96", TrainParams()), ("with SBS96", dataclasses.replace(TrainParams(), use_sbs96=True))):
        res = cross_validate(cohort.profiles, cohort.labels, cohort.patients, params, seed=7, tune_c=False, with_boosting=False)
        m = res.summary()["logistic"]["pooled"]
        print(f"   {label:<14} " + "  ".join(f"{k} {m[k]:.3f}" for k in KEYS))


def pairs(root: Path) -> None:
    print("\n4. Gene pairs (Fisher exact, BH) in two cohorts")
    checks = {
        "lgg": [("IDH1", "TP53"), ("TP53", "ATRX"), ("CIC", "FUBP1"), ("TP53", "CIC"), ("ATRX", "CIC")],
        "uvm": [("GNAQ", "GNA11"), ("BAP1", "SF3B1")],
    }
    for code, wanted in checks.items():
        study = load_study(root, code)
        genes = sorted({g for pair in wanted for g in pair})
        tests = {(t.gene_a, t.gene_b): t for t in pair_tests(study.profiles, genes=genes)}
        for a, b in wanted:
            t = tests.get((a, b)) or tests[(b, a)]
            d = t.as_dict()
            print(f"   {code.upper():<4} {a:<6} {b:<6} {d['tendency']:<19} OR {d['odds_ratio']!s:<7} q {d['q']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/pancan")
    args = parser.parse_args()
    root = Path(args.data)
    cohort = load_cohort(root, BENCHMARK)
    print(f"{len(cohort)} samples from {len(set(cohort.patients))} patients, {len(set(cohort.labels))} tumour types\n")
    tmb_agreement(cohort)
    cv_over_seeds(cohort)
    sbs96_ablation(cohort)
    pairs(root)


if __name__ == "__main__":
    main()
