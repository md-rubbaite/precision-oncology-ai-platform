from __future__ import annotations

import pytest

from oncoai.features import profiles
from oncoai.maf import read_maf
from oncoai.synthetic import label_of, maf_text

HEADER = "Hugo_Symbol\tTumor_Sample_Barcode\tVariant_Classification\tVariant_Type\tReference_Allele\tTumor_Seq_Allele2"


def maf(*rows: str, header: str = HEADER) -> str:
    return "\n".join([header, *rows]) + "\n"


@pytest.fixture(scope="session")
def synthetic_text() -> str:
    return maf_text(per_type=30, seed=3)


@pytest.fixture(scope="session")
def synthetic_profiles(synthetic_text):
    data = read_maf(synthetic_text)
    items = list(profiles(data.by_sample()).values())
    return items, [label_of(p.sample) for p in items]


@pytest.fixture(scope="session")
def trained(synthetic_profiles):
    from oncoai.training import train

    items, labels = synthetic_profiles
    model, report = train(items, labels, [p.sample for p in items], n_splits=3, seed=5, target_accuracy=0.9)
    return model, report
