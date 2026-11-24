"""Per-sample mutation profiles and the feature space used by the classifier.

A profile summarises one tumour: tumour mutational burden (TMB), the six single-base substitution
classes (SBS6), the 96 trinucleotide channels (SBS96, when the file has sequence context), and the
genes and protein changes hit by protein-changing mutations.

The feature space (which genes, which recurrent protein changes, how numbers are scaled) is learned
from training samples only, so cross-validation never sees the held-out fold. It is stored as plain
JSON next to the model.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .maf import Mutation

DEFAULT_EXOME_MB = 30.0  # cBioPortal divides the nonsynonymous count by 30 Mb for whole-exome data

_COMP = str.maketrans("ACGT", "TGCA")
SBS6 = ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
SBS96 = tuple(f"{five}[{sub}]{three}" for sub in SBS6 for five in "ACGT" for three in "ACGT")
_SBS96_INDEX = {name: i for i, name in enumerate(SBS96)}


def substitution(m: Mutation) -> tuple[str, str | None] | None:
    """Return (SBS6 class, SBS96 channel or None) for a single-base substitution, pyrimidine-centred."""
    if not m.is_snv or m.ref not in "ACGT" or m.alt not in "ACGT":
        return None
    ref, alt = m.ref, m.alt
    ctx = m.context
    five = three = None
    if ctx:
        c = len(ctx) // 2
        if ctx[c] == ref and c >= 1:
            five, three = ctx[c - 1], ctx[c + 1]
    if ref in "GA":
        ref, alt = ref.translate(_COMP), alt.translate(_COMP)
        if five is not None and three is not None:
            five, three = three.translate(_COMP), five.translate(_COMP)
    sub = f"{ref}>{alt}"
    known = five is not None and three is not None and five in "ACGT" and three in "ACGT"
    channel = f"{five}[{sub}]{three}" if known else None
    return sub, channel


@dataclass(slots=True)
class Profile:
    sample: str
    n_total: int = 0
    n_nonsyn: int = 0
    n_snv: int = 0
    n_indel: int = 0
    tmb: float = 0.0
    sbs6: list[int] = field(default_factory=lambda: [0] * 6)
    sbs96: list[int] = field(default_factory=lambda: [0] * 96)
    genes: set[str] = field(default_factory=set)
    changes: set[str] = field(default_factory=set)
    gene_changes: dict[str, list[str]] = field(default_factory=dict)

    @property
    def sbs6_fractions(self) -> list[float]:
        total = sum(self.sbs6)
        return [c / total if total else 0.0 for c in self.sbs6]

    @property
    def sbs96_fractions(self) -> list[float]:
        total = sum(self.sbs96)
        return [c / total if total else 0.0 for c in self.sbs96]

    @property
    def has_context(self) -> bool:
        return sum(self.sbs96) > 0

    def summary(self) -> dict:
        return {
            "sample": self.sample,
            "mutations": self.n_total,
            "nonsynonymous": self.n_nonsyn,
            "tmb_per_mb": round(self.tmb, 3),
            "snv": self.n_snv,
            "indel": self.n_indel,
            "sbs6": dict(zip(SBS6, [round(f, 4) for f in self.sbs6_fractions], strict=True)),
            "sbs96_available": self.has_context,
            "genes_hit": len(self.genes),
        }


def profile(sample: str, mutations: Iterable[Mutation], exome_mb: float = DEFAULT_EXOME_MB) -> Profile:
    p = Profile(sample=sample)
    for m in mutations:
        p.n_total += 1
        if m.vtype in ("INS", "DEL"):
            p.n_indel += 1
        sub = substitution(m)
        if sub is not None:
            p.n_snv += 1
            p.sbs6[SBS6.index(sub[0])] += 1
            if sub[1] is not None:
                p.sbs96[_SBS96_INDEX[sub[1]]] += 1
        if m.nonsynonymous:
            p.n_nonsyn += 1
            p.genes.add(m.gene)
            if m.protein and m.protein.startswith("p.") and m.protein not in ("p.=", "p.?"):
                change = f"{m.gene} {m.protein}"
                p.changes.add(change)
                p.gene_changes.setdefault(m.gene, []).append(m.protein)
    p.tmb = p.n_nonsyn / exome_mb if exome_mb > 0 else 0.0
    return p


def profiles(by_sample: dict[str, list[Mutation]], exome_mb: float = DEFAULT_EXOME_MB) -> dict[str, Profile]:
    return {s: profile(s, ms, exome_mb) for s, ms in sorted(by_sample.items())}


NUMERIC = ("log2_tmb", "indel_fraction", *(f"sbs6:{s}" for s in SBS6))


@dataclass(slots=True)
class FeatureSpace:
    """Which columns the model sees, learned from training data only."""

    genes: list[str]
    changes: list[str]
    use_sbs96: bool
    mean: list[float]
    scale: list[float]

    @property
    def numeric_names(self) -> list[str]:
        names = list(NUMERIC)
        if self.use_sbs96:
            names += [f"sbs96:{c}" for c in SBS96]
        return names

    @property
    def names(self) -> list[str]:
        return self.numeric_names + [f"gene:{g}" for g in self.genes] + [f"change:{c}" for c in self.changes]

    @classmethod
    def fit(
        cls,
        items: Sequence[Profile],
        labels: Sequence[str],
        *,
        min_gene_fraction: float = 0.03,
        max_genes: int = 300,
        min_change_count: int = 3,
        use_sbs96: bool = True,
    ) -> FeatureSpace:
        if len(items) != len(labels) or not items:
            raise ValueError("need one label per profile and at least one profile")
        per_class: dict[str, Counter[str]] = {}
        class_size: Counter[str] = Counter(labels)
        change_count: Counter[str] = Counter()
        for p, y in zip(items, labels, strict=True):
            per_class.setdefault(y, Counter()).update(p.genes)
            change_count.update(p.changes)
        best: dict[str, float] = {}
        for y, counts in per_class.items():
            for gene, n in counts.items():
                frac = n / class_size[y]
                if n >= 2 and frac >= min_gene_fraction:
                    best[gene] = max(best.get(gene, 0.0), frac)
        genes = sorted(best, key=lambda g: (-best[g], g))[:max_genes]
        changes = sorted(c for c, n in change_count.items() if n >= min_change_count)
        use_sbs96 = use_sbs96 and all(p.has_context for p in items if p.n_snv)
        space = cls(genes=sorted(genes), changes=changes, use_sbs96=use_sbs96, mean=[], scale=[])
        raw = np.array([space._numeric(p) for p in items], dtype=float)
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        std[std < 1e-9] = 1.0
        space.mean, space.scale = mean.tolist(), std.tolist()
        return space

    def _numeric(self, p: Profile) -> list[float]:
        values = [math.log2(1.0 + p.tmb), p.n_indel / p.n_total if p.n_total else 0.0, *p.sbs6_fractions]
        if self.use_sbs96:
            values += p.sbs96_fractions
        return values

    def transform(self, items: Sequence[Profile]) -> np.ndarray:
        n_num = len(self.numeric_names)
        gene_ix = {g: i for i, g in enumerate(self.genes)}
        change_ix = {c: i for i, c in enumerate(self.changes)}
        out = np.zeros((len(items), n_num + len(self.genes) + len(self.changes)), dtype=float)
        mean, scale = np.asarray(self.mean), np.asarray(self.scale)
        for row, p in enumerate(items):
            out[row, :n_num] = (np.asarray(self._numeric(p)) - mean) / scale
            for g in p.genes:
                if g in gene_ix:
                    out[row, n_num + gene_ix[g]] = 1.0
            for c in p.changes:
                if c in change_ix:
                    out[row, n_num + len(self.genes) + change_ix[c]] = 1.0
        return out

    def to_dict(self) -> dict:
        return {"genes": self.genes, "changes": self.changes, "use_sbs96": self.use_sbs96, "mean": self.mean, "scale": self.scale}

    @classmethod
    def from_dict(cls, data: dict) -> FeatureSpace:
        space = cls(
            genes=[str(g) for g in data["genes"]],
            changes=[str(c) for c in data["changes"]],
            use_sbs96=bool(data["use_sbs96"]),
            mean=[float(v) for v in data["mean"]],
            scale=[float(v) for v in data["scale"]],
        )
        if len(space.mean) != len(space.numeric_names) or len(space.scale) != len(space.numeric_names):
            raise ValueError("feature space scaling does not match its numeric features")
        if any(not math.isfinite(v) or v <= 0 for v in space.scale) or any(not math.isfinite(v) for v in space.mean):
            raise ValueError("feature space scaling must be finite and positive")
        return space
