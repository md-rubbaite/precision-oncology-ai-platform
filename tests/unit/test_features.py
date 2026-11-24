import numpy as np
import pytest

from oncoai.features import NUMERIC, SBS6, SBS96, FeatureSpace, profile, substitution
from oncoai.maf import Mutation


def snv(ref, alt, ctx="", gene="G", vclass="Missense_Mutation", protein="p.A1T", sample="S"):
    return Mutation(sample, gene, vclass, "SNP", ref, alt, protein=protein, context=ctx)


def test_sbs96_has_96_unique_channels_in_cosmic_order():
    assert len(SBS96) == 96 and len(set(SBS96)) == 96
    assert SBS96[0] == "A[C>A]A" and SBS96[-1] == "T[T>G]T"


@pytest.mark.parametrize(
    ("ref", "alt", "ctx", "sub", "channel"),
    [
        ("C", "T", "GGGGACGGGGG", "C>T", "A[C>T]G"),
        ("G", "A", "AAAATGCAAAA", "C>T", "G[C>T]A"),  # reverse complement: TGC -> GCA
        ("T", "G", "CCCCATCCCCC", "T>G", "A[T>G]C"),
        ("A", "C", "TTTTCAGTTTT", "T>G", "C[T>G]G"),  # CAG -> CTG
    ],
)
def test_substitution_is_pyrimidine_centred(ref, alt, ctx, sub, channel):
    assert substitution(snv(ref, alt, ctx)) == (sub, channel)


def test_substitution_without_or_with_wrong_context():
    assert substitution(snv("C", "T")) == ("C>T", None)
    assert substitution(snv("C", "T", "AAAAAGAAAAA")) == ("C>T", None)  # centre is not the reference base
    assert substitution(snv("C", "T", "AAAANCNAAAA")) == ("C>T", None)


def test_indels_and_multi_base_are_not_substitutions():
    assert substitution(Mutation("S", "G", "Frame_Shift_Del", "DEL", "A", "-")) is None
    assert substitution(Mutation("S", "G", "Missense_Mutation", "DNP", "AC", "GT")) is None


def test_profile_counts_and_tmb():
    muts = [
        snv("C", "T", "AAAAACGAAAA", gene="TP53", protein="p.R175H"),
        snv("G", "T", "AAAAAGAAAAA", gene="KRAS", protein="p.G12V"),
        snv("C", "T", gene="TTN", vclass="Silent", protein="p.A5="),
        Mutation("S", "ATRX", "Frame_Shift_Del", "DEL", "A", "-", protein="p.K1fs"),
    ]
    p = profile("S", muts, exome_mb=30)
    assert (p.n_total, p.n_nonsyn, p.n_snv, p.n_indel) == (4, 3, 3, 1)
    assert p.tmb == pytest.approx(3 / 30)
    assert p.genes == {"TP53", "KRAS", "ATRX"}
    assert "TP53 p.R175H" in p.changes and "KRAS p.G12V" in p.changes
    assert p.sbs6[SBS6.index("C>T")] == 2 and p.sbs6[SBS6.index("C>A")] == 1
    assert sum(p.sbs96) == 2 and p.has_context
    assert sum(p.sbs6_fractions) == pytest.approx(1.0)


def test_profile_ignores_uninformative_protein_strings():
    p = profile("S", [snv("C", "T", protein="p.?"), snv("C", "A", protein="")])
    assert p.changes == set() and p.genes == {"G"}


def test_summary_keys():
    s = profile("S", [snv("C", "T")]).summary()
    assert set(s) >= {"sample", "tmb_per_mb", "sbs6", "sbs96_available", "genes_hit"}
    assert s["sbs6"]["C>T"] == 1.0


def _items():
    a = [profile(f"A{i}", [snv("C", "T", gene="GA", protein="p.X1Y"), snv("C", "A", gene=f"R{i}")]) for i in range(10)]
    b = [profile(f"B{i}", [snv("T", "C", gene="GB", protein="p.Q2R"), snv("T", "G", gene=f"Q{i}")]) for i in range(10)]
    return a + b, ["A"] * 10 + ["B"] * 10


def test_feature_space_learns_only_recurrent_genes_and_changes():
    items, labels = _items()
    space = FeatureSpace.fit(items, labels, min_gene_fraction=0.3, min_change_count=3)
    assert space.genes == ["GA", "GB"]
    assert space.changes == ["GA p.X1Y", "GB p.Q2R"]
    assert space.names[: len(NUMERIC)] == list(NUMERIC)


def test_transform_scales_numeric_and_marks_binary_columns():
    items, labels = _items()
    space = FeatureSpace.fit(items, labels, min_gene_fraction=0.3, use_sbs96=False)
    x = space.transform(items)
    assert x.shape == (20, len(space.names))
    n = len(NUMERIC)
    assert np.allclose(x[:, :n].mean(axis=0), 0.0, atol=1e-9)
    assert x[0, n + space.genes.index("GA")] == 1.0 and x[0, n + space.genes.index("GB")] == 0.0


def test_unseen_genes_are_ignored_at_transform():
    items, labels = _items()
    space = FeatureSpace.fit(items, labels, min_gene_fraction=0.3)
    new = profile("N", [snv("C", "T", gene="NEVERSEEN")])
    assert space.transform([new])[0, len(space.numeric_names) :].sum() == 0


def test_sbs96_used_only_when_every_profile_has_context():
    items, labels = _items()
    assert FeatureSpace.fit(items, labels, use_sbs96=True).use_sbs96 is False  # _items have no context
    with_ctx = [profile(f"C{i}", [snv("C", "T", "AAAAACAAAAA", gene=f"Z{i}")]) for i in range(4)]
    space = FeatureSpace.fit(with_ctx, ["A", "A", "B", "B"], use_sbs96=True)
    assert space.use_sbs96 and len(space.numeric_names) == len(NUMERIC) + 96


def test_feature_space_round_trip_and_validation():
    items, labels = _items()
    space = FeatureSpace.fit(items, labels)
    again = FeatureSpace.from_dict(space.to_dict())
    assert np.allclose(again.transform(items), space.transform(items))
    bad = space.to_dict() | {"scale": [0.0] * len(space.scale)}
    with pytest.raises(ValueError, match="positive"):
        FeatureSpace.from_dict(bad)
    with pytest.raises(ValueError, match="does not match"):
        FeatureSpace.from_dict(space.to_dict() | {"mean": [0.0]})


def test_fit_requires_matching_labels():
    with pytest.raises(ValueError):
        FeatureSpace.fit([], [])
