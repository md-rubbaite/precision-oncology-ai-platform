"""Tumour-type classifier from a mutation profile: multinomial logistic regression.

Training uses scikit-learn. The trained model is stored as JSON (classes, feature space,
coefficients) and scored with NumPy, so serving never unpickles a file: a model file can be
read and reviewed by a person, and loading one cannot execute code (ADR 0001).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import FeatureSpace, Profile

FORMAT = "oncoai-model/1"
RESEARCH_USE = "Research use only. Not for diagnosis, prognosis or treatment decisions."
MAX_MODEL_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TrainParams:
    C: float = 1.0
    min_gene_fraction: float = 0.03
    max_genes: int = 300
    min_change_count: int = 3
    use_sbs96: bool = False
    class_weight: str | None = "balanced"
    max_iter: int = 2000

    def to_dict(self) -> dict:
        return {
            "C": self.C,
            "min_gene_fraction": self.min_gene_fraction,
            "max_genes": self.max_genes,
            "min_change_count": self.min_change_count,
            "use_sbs96": self.use_sbs96,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
        }


@dataclass(slots=True)
class Prediction:
    sample: str
    probabilities: dict[str, float]
    top: str
    confidence: float
    abstained: bool
    threshold: float
    reasons: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        ranked = sorted(self.probabilities.items(), key=lambda kv: -kv[1])
        return {
            "sample": self.sample,
            "call": None if self.abstained else self.top,
            "abstained": self.abstained,
            "most_likely": self.top,
            "confidence": round(self.confidence, 4),
            "abstain_below": self.threshold,
            "probabilities": [{"label": k, "p": round(v, 4)} for k, v in ranked],
            "reasons": self.reasons,
            "notice": RESEARCH_USE,
        }


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass(slots=True)
class Model:
    classes: list[str]
    space: FeatureSpace
    coef: np.ndarray  # (classes, features)
    intercept: np.ndarray  # (classes,)
    threshold: float = 0.0
    meta: dict = field(default_factory=dict)

    def decision(self, items: Sequence[Profile]) -> np.ndarray:
        x = self.space.transform(items)
        return x @ self.coef.T + self.intercept

    def predict_proba(self, items: Sequence[Profile]) -> np.ndarray:
        return softmax(self.decision(items))

    def explain(self, p: Profile, label: str, top: int = 5) -> list[dict]:
        """Largest contributions (coefficient x value) toward ``label`` relative to the class average.

        Contributions are reported against the mean over classes, because in a softmax model only
        differences between classes matter.
        """
        x = self.space.transform([p])[0]
        k = self.classes.index(label)
        rel = self.coef[k] - self.coef.mean(axis=0)
        contrib = rel * x
        names = self.space.names
        n_numeric = len(self.space.numeric_names)
        order = np.argsort(-contrib)
        out = []
        for i in order[:top]:
            if contrib[i] <= 0:
                break
            item = {"feature": names[i], "contribution": round(float(contrib[i]), 3)}
            if i < n_numeric:  # numeric features are standardised: say which side of the training mean the sample is
                item["direction"] = "higher" if x[i] > 0 else "lower"
            out.append(item)
        return out

    def predict(self, items: Sequence[Profile], explain: bool = True) -> list[Prediction]:
        proba = self.predict_proba(items)
        results = []
        for p, row in zip(items, proba, strict=True):
            k = int(np.argmax(row))
            label = self.classes[k]
            conf = float(row[k])
            results.append(
                Prediction(
                    sample=p.sample,
                    probabilities={c: float(v) for c, v in zip(self.classes, row, strict=True)},
                    top=label,
                    confidence=conf,
                    abstained=conf < self.threshold,
                    threshold=self.threshold,
                    reasons=self.explain(p, label) if explain else [],
                )
            )
        return results

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "notice": RESEARCH_USE,
            "classes": self.classes,
            "threshold": self.threshold,
            "feature_space": self.space.to_dict(),
            "coef": self.coef.round(8).tolist(),
            "intercept": self.intercept.round(8).tolist(),
            "meta": self.meta,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> Model:
        if data.get("format") != FORMAT:
            raise ValueError(f"not an {FORMAT} file")
        classes = [str(c) for c in data["classes"]]
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("model needs at least two distinct classes")
        space = FeatureSpace.from_dict(data["feature_space"])
        coef = np.asarray(data["coef"], dtype=float)
        intercept = np.asarray(data["intercept"], dtype=float)
        if coef.shape != (len(classes), len(space.names)) or intercept.shape != (len(classes),):
            raise ValueError("coefficient shape does not match classes and features")
        if not (np.isfinite(coef).all() and np.isfinite(intercept).all()):
            raise ValueError("coefficients must be finite")
        threshold = float(data.get("threshold", 0.0))
        if not (0.0 <= threshold <= 1.0) or math.isnan(threshold):
            raise ValueError("threshold must be between 0 and 1")
        return cls(classes=classes, space=space, coef=coef, intercept=intercept, threshold=threshold, meta=dict(data.get("meta", {})))

    @classmethod
    def load(cls, path: Path) -> Model:
        if path.stat().st_size > MAX_MODEL_BYTES:
            raise ValueError("model file is larger than allowed")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def fit(items: Sequence[Profile], labels: Sequence[str], params: TrainParams = TrainParams()) -> Model:
    """Fit the feature space and the classifier on training data only."""
    from sklearn.linear_model import LogisticRegression

    space = FeatureSpace.fit(
        items,
        labels,
        min_gene_fraction=params.min_gene_fraction,
        max_genes=params.max_genes,
        min_change_count=params.min_change_count,
        use_sbs96=params.use_sbs96,
    )
    x = space.transform(items)
    clf = LogisticRegression(C=params.C, class_weight=params.class_weight, max_iter=params.max_iter)
    clf.fit(x, list(labels))
    classes = [str(c) for c in clf.classes_]
    coef, intercept = clf.coef_, clf.intercept_
    if len(classes) == 2:  # binary sklearn models keep one row; expand to the softmax form
        coef = np.vstack([-coef[0] / 2, coef[0] / 2])
        intercept = np.array([-intercept[0] / 2, intercept[0] / 2])
    return Model(classes=classes, space=space, coef=np.asarray(coef, float), intercept=np.asarray(intercept, float))
