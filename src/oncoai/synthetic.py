"""Synthetic mutation cohorts for tests, CI and the offline demo.

Everything here is invented: tumour types are called SYNA, SYNB, ...; genes are SG001, SG002, ...;
the driver frequencies, hotspots and substitution spectra are arbitrary. Nothing in this module
describes real tumour biology. It exists so the pipeline can be exercised without downloading data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .maf import REQUIRED

_BASES = "ACGT"
_COMP = str.maketrans("ACGT", "TGCA")
_SUBS = ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
_CLASSES = ("Missense_Mutation", "Missense_Mutation", "Missense_Mutation", "Silent", "Nonsense_Mutation", "Splice_Site", "3'UTR")
COLUMNS = (*REQUIRED, "Chromosome", "Start_Position", "HGVSp_Short", "t_ref_count", "t_alt_count", "CONTEXT")


@dataclass(frozen=True, slots=True)
class SyntheticType:
    code: str
    tmb_median: float  # nonsynonymous per Mb, before noise
    spectrum: tuple[float, ...]  # weights over C>A, C>G, C>T, T>A, T>C, T>G
    drivers: tuple[tuple[str, float, str], ...]  # (gene, probability, hotspot or "")
    indel_fraction: float = 0.08


TYPES = (
    SyntheticType("SYNA", 1.0, (1, 1, 6, 1, 2, 1), (("SG001", 0.7, "p.R175H"), ("SG002", 0.4, ""), ("SG010", 0.2, ""))),
    SyntheticType("SYNB", 0.6, (1, 1, 3, 1, 3, 1), (("SG003", 0.8, "p.R132H"), ("SG001", 0.5, ""), ("SG004", 0.3, ""))),
    SyntheticType("SYNC", 4.0, (5, 1, 2, 1, 1, 1), (("SG005", 0.5, "p.G12D"), ("SG001", 0.6, ""), ("SG006", 0.35, ""))),
    SyntheticType("SYND", 0.4, (1, 2, 2, 1, 2, 2), (("SG007", 0.85, "p.Q209L"), ("SG008", 0.3, ""))),
    SyntheticType("SYNE", 2.0, (1, 1, 2, 3, 1, 3), (("SG009", 0.5, "p.V600E"), ("SG002", 0.3, ""), ("SG011", 0.3, "")), 0.2),
)
GENES = tuple(f"SG{i:03d}" for i in range(1, 401))


def _context(rng: random.Random, ref: str) -> str:
    left = "".join(rng.choice(_BASES) for _ in range(5))
    right = "".join(rng.choice(_BASES) for _ in range(5))
    return f"{left}{ref}{right}"


def _snv(rng: random.Random, spectrum: tuple[float, ...]) -> tuple[str, str]:
    sub = rng.choices(_SUBS, weights=spectrum)[0]
    ref, alt = sub[0], sub[2]
    if rng.random() < 0.5:  # report half on the other strand, as real files do
        ref, alt = ref.translate(_COMP), alt.translate(_COMP)
    return ref, alt


def generate_rows(
    per_type: int = 40, seed: int = 1, types: tuple[SyntheticType, ...] = TYPES, exome_mb: float = 30.0
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for t in types:
        for i in range(per_type):
            sample = f"SYN-{t.code[-1]}{i:04d}-01"
            n_nonsyn = max(1, round(rng.lognormvariate(0, 0.6) * t.tmb_median * exome_mb))
            pos = 1000
            muts: list[tuple[str, str, str]] = []  # gene, class, protein
            for gene, prob, hotspot in t.drivers:
                if rng.random() < prob:
                    protein = (
                        hotspot if hotspot and rng.random() < 0.8 else f"p.{rng.choice('ACDEFGHIKLMNPQRSTVWY')}{rng.randint(20, 900)}*"
                    )
                    muts.append((gene, "Missense_Mutation" if not protein.endswith("*") else "Nonsense_Mutation", protein))
            while len(muts) < n_nonsyn * 1.4:
                vclass = rng.choice(_CLASSES)
                protein = (
                    ""
                    if vclass in ("Silent", "3'UTR")
                    else f"p.{rng.choice('ACDEFGHIKLMNPQRSTVWY')}{rng.randint(20, 900)}{rng.choice('ACDEFGHIKLMNPQRSTVWY')}"
                )
                muts.append((rng.choice(GENES), vclass, protein))
            for gene, vclass, protein in muts:
                pos += rng.randint(50, 5000)
                if rng.random() < t.indel_fraction and vclass != "Silent":
                    vtype, ref, alt = "DEL", rng.choice(_BASES), "-"
                    vclass = "Frame_Shift_Del"
                    protein = f"p.K{rng.randint(20, 900)}fs"
                else:
                    vtype = "SNP"
                    ref, alt = _snv(rng, t.spectrum)
                depth = rng.randint(30, 200)
                alt_reads = max(3, round(depth * rng.uniform(0.1, 0.5)))
                rows.append(
                    {
                        "Hugo_Symbol": gene,
                        "Tumor_Sample_Barcode": sample,
                        "Variant_Classification": vclass,
                        "Variant_Type": vtype,
                        "Reference_Allele": ref,
                        "Tumor_Seq_Allele2": alt,
                        "Chromosome": str(rng.randint(1, 22)),
                        "Start_Position": str(pos),
                        "HGVSp_Short": protein,
                        "t_ref_count": str(depth - alt_reads),
                        "t_alt_count": str(alt_reads),
                        "CONTEXT": _context(rng, ref) if vtype == "SNP" else "",
                    }
                )
    return rows


def to_maf(rows: list[dict[str, str]]) -> str:
    lines = ["#synthetic data generated by oncoai.synthetic; not real patients", "\t".join(COLUMNS)]
    lines += ["\t".join(r.get(c, "") for c in COLUMNS) for r in rows]
    return "\n".join(lines) + "\n"


def maf_text(per_type: int = 40, seed: int = 1) -> str:
    return to_maf(generate_rows(per_type, seed))


def label_of(sample: str) -> str:
    """Synthetic sample ids carry their type letter: SYN-A0001-01 -> SYNA."""
    return "SYN" + sample.split("-")[1][0]
