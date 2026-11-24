"""Cohort statistics: how often genes are hit, and which gene pairs co-occur or exclude each other.

The pairwise test is Fisher's exact test with Benjamini-Hochberg correction. It treats every sample
as equally likely to carry a mutation, which is not true: hypermutated tumours carry many mutations
in every gene. That inflates apparent co-occurrence and hides exclusivity, so results are labelled
as screening output. Tests that model per-sample rates (for example DISCOVER) are the better choice
for conclusions; see docs/architecture/overview.md.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from .features import Profile


def _logc(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Two-sided Fisher exact test for [[a, b], [c, d]]. Returns (odds ratio, p-value).

    a = both mutated, b = only first, c = only second, d = neither. The p-value sums the
    probabilities of all tables with the same margins that are no more likely than the observed one.
    """
    if min(a, b, c, d) < 0:
        raise ValueError("counts must be non-negative")
    row1, col1, n = a + b, a + c, a + b + c + d
    lo, hi = max(0, row1 + col1 - n), min(row1, col1)

    def logp(x: int) -> float:
        return _logc(col1, x) + _logc(n - col1, row1 - x) - _logc(n, row1)

    observed = logp(a)
    p = sum(math.exp(lp) for lp in (logp(x) for x in range(lo, hi + 1)) if lp <= observed + 1e-7)
    odds = (a * d) / (b * c) if b * c else (math.inf if a * d > 0 else math.nan)
    return odds, min(1.0, p)


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    q = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * m / rank)
        q[i] = running
    return q


@dataclass(frozen=True, slots=True)
class GeneFrequency:
    gene: str
    samples: int
    fraction: float
    top_changes: list[tuple[str, int]]


def recurrence(items: Sequence[Profile], top: int = 20) -> list[GeneFrequency]:
    """Genes hit by protein-changing mutations, by number of samples."""
    counts: Counter[str] = Counter()
    changes: dict[str, Counter[str]] = {}
    for p in items:
        counts.update(p.genes)
        for gene, prots in p.gene_changes.items():
            changes.setdefault(gene, Counter()).update(set(prots))
    n = len(items) or 1
    out = []
    for gene, k in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]:
        out.append(GeneFrequency(gene, k, round(k / n, 4), changes.get(gene, Counter()).most_common(3)))
    return out


@dataclass(frozen=True, slots=True)
class PairTest:
    gene_a: str
    gene_b: str
    both: int
    only_a: int
    only_b: int
    neither: int
    odds_ratio: float
    p_value: float
    q_value: float

    @property
    def tendency(self) -> str:
        if math.isnan(self.odds_ratio):
            return "undetermined"
        return "co-occurrence" if self.odds_ratio > 1 else "mutual exclusivity"

    def as_dict(self) -> dict:
        odds = self.odds_ratio
        return {
            "genes": [self.gene_a, self.gene_b],
            "table": {"both": self.both, "only_a": self.only_a, "only_b": self.only_b, "neither": self.neither},
            "odds_ratio": None if math.isnan(odds) else ("inf" if math.isinf(odds) else round(odds, 3)),
            "p": float(f"{self.p_value:.3g}"),
            "q": float(f"{self.q_value:.3g}"),
            "tendency": self.tendency,
        }


def pair_tests(
    items: Sequence[Profile], genes: Sequence[str] | None = None, top: int = 12, exclude_hypermutated_above: float | None = None
) -> list[PairTest]:
    """Test every pair among ``genes`` (default: the ``top`` most frequently hit genes).

    ``exclude_hypermutated_above`` drops samples with TMB above that value (mutations per Mb)
    before testing, a crude guard against the rate problem described in the module docstring.
    """
    pool = [p for p in items if exclude_hypermutated_above is None or p.tmb <= exclude_hypermutated_above]
    if genes is None:
        genes = [g.gene for g in recurrence(pool, top)]
    results = []
    for a_gene, b_gene in combinations(genes, 2):
        a = b = c = d = 0
        for p in pool:
            ha, hb = a_gene in p.genes, b_gene in p.genes
            if ha and hb:
                a += 1
            elif ha:
                b += 1
            elif hb:
                c += 1
            else:
                d += 1
        odds, pv = fisher_exact(a, b, c, d)
        results.append((a_gene, b_gene, a, b, c, d, odds, pv))
    q = benjamini_hochberg([r[7] for r in results])
    tests = [PairTest(*r, q_value=qv) for r, qv in zip(results, q, strict=True)]
    return sorted(tests, key=lambda t: (t.q_value, t.p_value, t.gene_a, t.gene_b))


def summarise(items: Sequence[Profile], labels: Sequence[str], top_genes: int = 8) -> dict:
    """Per-label sample counts, TMB distribution and most frequently hit genes."""
    out = {}
    for label in sorted(set(labels)):
        group = [p for p, y in zip(items, labels, strict=True) if y == label]
        tmbs = sorted(p.tmb for p in group)

        def q(f: float, tmbs: list[float] = tmbs) -> float:
            i = (len(tmbs) - 1) * f
            lo, hi = math.floor(i), math.ceil(i)
            return tmbs[lo] + (tmbs[hi] - tmbs[lo]) * (i - lo)

        out[label] = {
            "samples": len(group),
            "tmb": {"p10": round(q(0.1), 3), "median": round(q(0.5), 3), "p90": round(q(0.9), 3)},
            "top_genes": [{"gene": g.gene, "fraction": g.fraction} for g in recurrence(group, top_genes)],
        }
    return out
