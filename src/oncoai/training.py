"""Train the model that is served: evaluate first, then fit on all data with the chosen settings."""

from __future__ import annotations

import dataclasses
import platform
from collections import Counter
from collections.abc import Sequence

import numpy as np

from . import __version__
from .cohort import summarise
from .evaluate import abstention_table, choose_c, choose_threshold, cross_validate
from .features import Profile
from .model import RESEARCH_USE, Model, TrainParams, fit


def train(
    items: Sequence[Profile],
    labels: Sequence[str],
    groups: Sequence[str],
    params: TrainParams = TrainParams(),
    *,
    n_splits: int = 5,
    seed: int = 7,
    target_accuracy: float = 0.9,
    with_boosting: bool = False,
    data_source: dict | None = None,
) -> tuple[Model, dict]:
    """Return the final model and the evaluation report that justifies it.

    The abstention threshold is the lowest confidence at which out-of-fold accuracy on answered
    samples reaches ``target_accuracy``; the report shows how many samples that leaves unanswered.
    """
    cv = cross_validate(items, labels, groups, params, n_splits=n_splits, seed=seed, with_boosting=with_boosting)
    threshold = choose_threshold(cv.y_true, cv.proba["logistic"], cv.classes, target_accuracy)
    final_c = choose_c(items, labels, groups, params, seed)
    model = fit(items, labels, dataclasses.replace(params, C=final_c))
    model.threshold = threshold

    summary = cv.summary()
    answered = cv.proba["logistic"].max(axis=1) >= threshold
    correct = cv.proba["logistic"].argmax(axis=1) == np.array([cv.classes.index(y) for y in cv.y_true])
    report = {
        "notice": RESEARCH_USE,
        "samples": len(items),
        "patients": len(set(groups)),
        "labels": dict(sorted(Counter(labels).items())),
        "cross_validation": {
            "scheme": f"{len(cv.fold_sizes)}-fold stratified, grouped by patient; C chosen by inner 3-fold CV on log loss",
            "seed": seed,
            "chosen_c_per_fold": cv.chosen_c,
            "methods": summary,
        },
        "abstention": {
            "target_accuracy": target_accuracy,
            "threshold": threshold,
            "out_of_fold_coverage": round(float(answered.mean()), 4),
            "out_of_fold_accuracy_when_answered": round(float(correct[answered].mean()), 4) if answered.any() else None,
            "table": abstention_table(cv.y_true, cv.proba["logistic"], cv.classes),
        },
        "final_model": {
            "C": final_c,
            "features": len(model.space.names),
            "genes": len(model.space.genes),
            "protein_changes": len(model.space.changes),
        },
        "params": dataclasses.replace(params, C=final_c).to_dict(),
    }
    logistic = summary["logistic"]["pooled"]
    model.meta = {
        "oncoai_version": __version__,
        "python": platform.python_version(),
        "data_source": data_source or {},
        "training_labels": report["labels"],
        "cv": {k: logistic[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy", "log_loss", "brier", "ece")},
        "cv_per_class": logistic["per_class"],
        "abstention": {k: report["abstention"][k] for k in ("threshold", "out_of_fold_coverage", "out_of_fold_accuracy_when_answered")},
        "cohort": summarise(items, labels),
        "params": report["params"],
    }
    return model, report
