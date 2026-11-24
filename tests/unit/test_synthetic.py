from oncoai.maf import read_maf
from oncoai.synthetic import GENES, TYPES, generate_rows, label_of, maf_text


def test_generation_is_deterministic_and_labelled():
    assert maf_text(per_type=3, seed=1) == maf_text(per_type=3, seed=1)
    assert maf_text(per_type=3, seed=1) != maf_text(per_type=3, seed=2)
    data = read_maf(maf_text(per_type=3, seed=1))
    samples = {m.sample for m in data.mutations}
    assert len(samples) == 3 * len(TYPES)
    assert {label_of(s) for s in samples} == {t.code for t in TYPES}


def test_synthetic_rows_pass_every_check_and_use_invented_names():
    text = maf_text(per_type=5, seed=7)
    data = read_maf(text)
    assert data.qc.kept == data.qc.rows and not data.qc.rejected and not data.qc.warnings
    assert text.startswith("#synthetic")
    drivers = {g for t in TYPES for g, _, _ in t.drivers}
    assert {m.gene for m in data.mutations} <= set(GENES) | drivers
    assert all(g.startswith("SG") for g in drivers)


def test_context_centre_matches_reference():
    for r in generate_rows(per_type=2, seed=3):
        if r["CONTEXT"]:
            assert r["CONTEXT"][5] == r["Reference_Allele"]
