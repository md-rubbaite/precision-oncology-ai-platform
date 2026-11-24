import math
import random

import pytest
from scipy.stats import false_discovery_control
from scipy.stats import fisher_exact as scipy_fisher

from oncoai.cohort import benjamini_hochberg, fisher_exact, pair_tests, recurrence, summarise
from oncoai.features import Profile


def test_fisher_matches_scipy_on_random_tables():
    rng = random.Random(4)
    for _ in range(300):
        a, b, c, d = (rng.randint(0, 25) for _ in range(4))
        odds, p = fisher_exact(a, b, c, d)
        ref = scipy_fisher([[a, b], [c, d]])
        assert p == pytest.approx(ref.pvalue, rel=1e-6, abs=1e-12)
        if b * c:
            assert odds == pytest.approx(ref.statistic)


def test_fisher_edge_cases():
    assert fisher_exact(0, 0, 0, 0)[1] == 1.0
    assert math.isinf(fisher_exact(5, 0, 0, 5)[0])
    assert fisher_exact(0, 5, 5, 0)[0] == 0.0
    with pytest.raises(ValueError):
        fisher_exact(-1, 0, 0, 0)


def test_benjamini_hochberg_matches_scipy():
    rng = random.Random(9)
    p = [rng.random() ** 3 for _ in range(50)]
    assert benjamini_hochberg(p) == pytest.approx(list(false_discovery_control(p)))


def _p(sample, genes, tmb=1.0, changes=None):
    pr = Profile(sample=sample, genes=set(genes), tmb=tmb)
    pr.gene_changes = changes or {}
    return pr


def test_recurrence_orders_by_samples_then_name():
    items = [_p("1", ["B", "A"], changes={"A": ["p.X1Y"]}), _p("2", ["A"], changes={"A": ["p.X1Y"]}), _p("3", ["C", "B"])]
    rec = recurrence(items, top=3)
    assert [(g.gene, g.samples) for g in rec] == [("A", 2), ("B", 2), ("C", 1)]
    assert rec[0].top_changes == [("p.X1Y", 2)]
    assert rec[0].fraction == pytest.approx(2 / 3, abs=1e-4)


def test_pair_tests_find_exclusivity_and_cooccurrence():
    items = [_p(f"a{i}", ["X", "Z"]) for i in range(20)] + [_p(f"b{i}", ["Y"]) for i in range(20)] + [_p(f"c{i}", []) for i in range(20)]
    tests = {(t.gene_a, t.gene_b): t for t in pair_tests(items, genes=["X", "Y", "Z"])}
    assert tests[("X", "Y")].tendency == "mutual exclusivity" and tests[("X", "Y")].q_value < 1e-3
    assert tests[("X", "Z")].tendency == "co-occurrence"
    d = tests[("X", "Z")].as_dict()
    assert d["odds_ratio"] == "inf" and d["table"]["both"] == 20


def test_pair_tests_can_leave_out_hypermutated_samples():
    items = [_p(f"h{i}", ["X", "Y"], tmb=50) for i in range(10)] + [_p(f"n{i}", ["X"] if i % 2 else ["Y"]) for i in range(10)]
    everyone = pair_tests(items, genes=["X", "Y"])[0]
    filtered = pair_tests(items, genes=["X", "Y"], exclude_hypermutated_above=10)[0]
    assert everyone.both == 10 and filtered.both == 0
    assert filtered.tendency == "mutual exclusivity"


def test_summarise_reports_tmb_quantiles_and_top_genes():
    items = [_p(str(i), ["A"], tmb=float(i)) for i in range(11)]
    s = summarise(items, ["T"] * 11)["T"]
    assert s["samples"] == 11 and s["tmb"] == {"p10": 1.0, "median": 5.0, "p90": 9.0}
    assert s["top_genes"][0] == {"gene": "A", "fraction": 1.0}
