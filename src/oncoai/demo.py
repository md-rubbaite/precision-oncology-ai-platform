"""Offline demo on synthetic data: read, check, profile, evaluate, train, predict, save, reload."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .cohort import pair_tests, recurrence
from .features import profiles
from .maf import read_maf
from .model import RESEARCH_USE, Model
from .synthetic import label_of, maf_text
from .training import train


def run(out: Path | None = None) -> int:
    print(f"oncoai demo (synthetic data only). {RESEARCH_USE}\n")

    print("1. Generate and read a synthetic cohort (5 invented tumour types, 40 samples each)")
    text = maf_text(per_type=40, seed=1)
    bad = "\nSG001\tSYN-A9999-01\tMissense_Mutation\tSNP\tC\tC\t1\t10\tp.X1X\t10\t5\tAAAAACAAAAA\n"
    data = read_maf(text + bad)
    qc = data.qc
    print(f"   {qc.rows} rows, {qc.kept} kept, {qc.samples} samples; rejected: {dict(qc.rejected)}")

    items = list(profiles(data.by_sample()).values())
    labels = [label_of(p.sample) for p in items]
    print("\n2. Profiles (first sample)")
    s = items[0].summary()
    print(f"   {s['sample']}: TMB {s['tmb_per_mb']}/Mb, {s['snv']} SNVs, C>T {s['sbs6']['C>T']:.0%}")

    print("\n3. Cohort statistics for SYNA")
    group = [p for p, y in zip(items, labels, strict=True) if y == "SYNA"]
    for g in recurrence(group, 3):
        print(f"   {g.gene} in {g.fraction:.0%} of samples")
    tests = pair_tests(group, top=6)
    significant = [t for t in tests if t.q_value < 0.05]
    print(f"   {len(tests)} gene pairs tested, {len(significant)} with q < 0.05 (drivers were drawn independently, so none expected)")

    print("\n4. Grouped cross-validation, then the final model")
    model, report = train(items, labels, [p.sample for p in items], n_splits=5, seed=7, target_accuracy=0.95)
    m = report["cross_validation"]["methods"]
    print(f"   logistic: accuracy {m['logistic']['pooled']['accuracy']:.2f}, ECE {m['logistic']['pooled']['ece']:.3f}")
    process, prior = m["mutational_process_only"]["pooled"]["accuracy"], m["class_prior"]["pooled"]["accuracy"]
    print(f"   process-only baseline: {process:.2f}; class prior: {prior:.2f}")
    a = report["abstention"]
    coverage, correct = a["out_of_fold_coverage"], a["out_of_fold_accuracy_when_answered"]
    print(f"   abstain below {a['threshold']:.2f}: answers {coverage:.0%}, {correct:.0%} correct")

    print("\n5. Save as JSON, reload, predict")
    path = out or Path(tempfile.mkdtemp()) / "demo-model.json"
    model.save(path)
    reloaded = Model.load(path)
    pred = reloaded.predict(items[:1])[0].as_dict()
    reasons = ", ".join(r["feature"] for r in pred["reasons"][:3])
    print(f"   {pred['sample']}: {pred['most_likely']} p={pred['confidence']:.2f} ({reasons})")
    print(f"   model file: {path}")
    print("\nDone. For the real benchmark: oncoai fetch && oncoai train --report report.json")
    return 0
