"""Honest evaluation: patient-grouped cross-validation, nested choice of regularisation, baselines,
calibration and an abstention threshold chosen from out-of-fold predictions.

Everything that is learned from data (gene list, recurrent protein changes, scaling, C) is learned
inside each training fold, so held-out samples never influence the model that scores them.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .features import Profile
from .model import Model, TrainParams, fit

C_GRID = (0.1, 0.3, 1.0, 3.0)


# ---------------------------------------------------------------- metrics
def metrics(y_true: Sequence[str], proba: np.ndarray, classes: Sequence[str]) -> dict:
    classes = list(classes)
    ix = {c: i for i, c in enumerate(classes)}
    y = np.array([ix[v] for v in y_true])
    n, k = proba.shape
    pred = proba.argmax(axis=1)
    acc = float((pred == y).mean())
    per_class = {}
    recalls, f1s = [], []
    for c, i in ix.items():
        tp = int(((pred == i) & (y == i)).sum())
        fp = int(((pred == i) & (y != i)).sum())
        support = int((y == i).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "support": support}
        if support:
            recalls.append(rec)
            f1s.append(f1)
    top3 = np.argsort(-proba, axis=1)[:, : min(3, k)]
    onehot = np.zeros_like(proba)
    onehot[np.arange(n), y] = 1.0
    eps = 1e-15
    confusion = np.zeros((k, k), dtype=int)
    for t, p in zip(y, pred, strict=True):
        confusion[t, p] += 1
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(float(np.mean(recalls)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "top3_accuracy": round(float(np.mean([y[i] in top3[i] for i in range(n)])), 4),
        "log_loss": round(float(-np.mean(np.log(np.clip(proba[np.arange(n), y], eps, 1.0)))), 4),
        "brier": round(float(np.mean(np.sum((proba - onehot) ** 2, axis=1))), 4),
        "ece": round(expected_calibration_error(proba, y), 4),
        "per_class": per_class,
        "confusion": {"labels": classes, "matrix": confusion.tolist()},
    }


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Top-label ECE with equal-width confidence bins."""
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def abstention_table(
    y_true: Sequence[str], proba: np.ndarray, classes: Sequence[str], thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
) -> list[dict]:
    ix = {c: i for i, c in enumerate(classes)}
    y = np.array([ix[v] for v in y_true])
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y
    rows = []
    for t in thresholds:
        answered = conf >= t
        cov = float(answered.mean())
        acc = float(correct[answered].mean()) if answered.any() else float("nan")
        rows.append({"threshold": t, "coverage": round(cov, 4), "accuracy_when_answered": None if math.isnan(acc) else round(acc, 4)})
    return rows


def choose_threshold(y_true: Sequence[str], proba: np.ndarray, classes: Sequence[str], target: float) -> float:
    """Lowest confidence threshold at which out-of-fold accuracy on answered samples reaches ``target``."""
    ix = {c: i for i, c in enumerate(classes)}
    y = np.array([ix[v] for v in y_true])
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y
    for t in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        answered = conf >= t
        if answered.sum() >= 10 and correct[answered].mean() >= target:
            return float(t)
    return 1.0


# ---------------------------------------------------------------- folds
def grouped_folds(labels: Sequence[str], groups: Sequence[str], n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    smallest = min(Counter(labels).values())
    n_splits = max(2, min(n_splits, smallest))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(labels))
    return [(tr, te) for tr, te in splitter.split(idx, list(labels), list(groups))]


def _subset(items, index):
    return [items[i] for i in index]


def choose_c(
    items: Sequence[Profile], labels: Sequence[str], groups: Sequence[str], params: TrainParams, seed: int, inner: int = 3
) -> float:
    """Pick C by inner grouped CV on log loss (training data only)."""
    best_c, best = params.C, math.inf
    folds = grouped_folds(labels, groups, inner, seed + 1)
    for c in C_GRID:
        losses = []
        for tr, te in folds:
            model = fit(_subset(items, tr), _subset(labels, tr), dataclasses.replace(params, C=c))
            proba = _aligned(model, _subset(items, te), model.classes)
            m = metrics(_subset(labels, te), proba, model.classes) if set(_subset(labels, te)) <= set(model.classes) else None
            if m:
                losses.append(m["log_loss"])
        if losses and np.mean(losses) < best:
            best, best_c = float(np.mean(losses)), c
    return best_c


def _aligned(model: Model, items: Sequence[Profile], classes: Sequence[str]) -> np.ndarray:
    raw = model.predict_proba(items)
    out = np.zeros((len(items), len(classes)))
    for j, c in enumerate(model.classes):
        out[:, list(classes).index(c)] = raw[:, j]
    return out


@dataclass(slots=True)
class CvResult:
    classes: list[str]
    y_true: list[str]
    proba: dict[str, np.ndarray]  # method -> out-of-fold probabilities
    chosen_c: list[float]
    fold_sizes: list[int]
    fold_metrics: dict[str, list[dict]] = field(default_factory=dict)

    def summary(self) -> dict:
        out = {}
        for name, proba in self.proba.items():
            m = metrics(self.y_true, proba, self.classes)
            folds = self.fold_metrics.get(name, [])
            spread = {}
            for key in ("accuracy", "balanced_accuracy", "macro_f1"):
                vals = [f[key] for f in folds]
                if vals:
                    spread[key] = {"mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals)), 4)}
            out[name] = {"pooled": m, "across_folds": spread}
        return out


def cross_validate(
    items: Sequence[Profile],
    labels: Sequence[str],
    groups: Sequence[str],
    params: TrainParams = TrainParams(),
    n_splits: int = 5,
    seed: int = 7,
    tune_c: bool = True,
    with_boosting: bool = True,
) -> CvResult:
    classes = sorted(set(labels))
    n = len(items)
    proba = {
        "logistic": np.zeros((n, len(classes))),
        "mutational_process_only": np.zeros((n, len(classes))),
        "class_prior": np.zeros((n, len(classes))),
    }
    if with_boosting:
        proba["gradient_boosting"] = np.zeros((n, len(classes)))
    fold_metrics: dict[str, list[dict]] = {k: [] for k in proba}
    chosen, sizes = [], []
    process_only = dataclasses.replace(params, max_genes=0, min_change_count=10**9)

    for f, (tr, te) in enumerate(grouped_folds(labels, groups, n_splits, seed)):
        tr_items, tr_labels, tr_groups = _subset(items, tr), _subset(labels, tr), _subset(groups, tr)
        te_items, te_labels = _subset(items, te), _subset(labels, te)
        c = choose_c(tr_items, tr_labels, tr_groups, params, seed + f) if tune_c else params.C
        chosen.append(c)
        sizes.append(len(te))

        model = fit(tr_items, tr_labels, dataclasses.replace(params, C=c))
        proba["logistic"][te] = _aligned(model, te_items, classes)

        base = fit(tr_items, tr_labels, dataclasses.replace(process_only, C=c))
        proba["mutational_process_only"][te] = _aligned(base, te_items, classes)

        prior = Counter(tr_labels)
        proba["class_prior"][te] = np.array([[prior[c_] / len(tr_labels) for c_ in classes]] * len(te))

        if with_boosting:
            from sklearn.ensemble import HistGradientBoostingClassifier

            space = model.space
            gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, class_weight="balanced", random_state=seed)
            gb.fit(space.transform(tr_items), tr_labels)
            raw = gb.predict_proba(space.transform(te_items))
            out = np.zeros((len(te), len(classes)))
            for j, cl in enumerate(gb.classes_):
                out[:, classes.index(str(cl))] = raw[:, j]
            proba["gradient_boosting"][te] = out

        for name in proba:
            fold_metrics[name].append(metrics(te_labels, proba[name][te], classes))

    return CvResult(classes=classes, y_true=list(labels), proba=proba, chosen_c=chosen, fold_sizes=sizes, fold_metrics=fold_metrics)
