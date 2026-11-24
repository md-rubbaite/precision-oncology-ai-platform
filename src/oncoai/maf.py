"""Read and check somatic mutation files in MAF format (as published by cBioPortal and the GDC).

Only the columns this platform uses are read. Every row is checked; rows that fail a check are
kept out of the analysis and reported with the reason, so a reader can see what was dropped.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

REQUIRED = (
    "Hugo_Symbol",
    "Tumor_Sample_Barcode",
    "Variant_Classification",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
)
OPTIONAL = ("Chromosome", "Start_Position", "HGVSp_Short", "t_ref_count", "t_alt_count", "CONTEXT")

# Variant classes that change the protein. Splice_Region is included because that is what
# reproduces cBioPortal's published TMB most closely (see docs/data/validation.md).
NONSYNONYMOUS = frozenset(
    {
        "Missense_Mutation",
        "Nonsense_Mutation",
        "Frame_Shift_Del",
        "Frame_Shift_Ins",
        "In_Frame_Del",
        "In_Frame_Ins",
        "Splice_Site",
        "Splice_Region",
        "Nonstop_Mutation",
        "Translation_Start_Site",
    }
)
KNOWN_CLASSES = NONSYNONYMOUS | {
    "Silent",
    "Intron",
    "3'UTR",
    "5'UTR",
    "3'Flank",
    "5'Flank",
    "IGR",
    "RNA",
    "lincRNA",
    "De_novo_Start_InFrame",
    "De_novo_Start_OutOfFrame",
    "Targeted_Region",
}
VARIANT_TYPES = frozenset({"SNP", "DNP", "TNP", "ONP", "INS", "DEL"})
_ALLELE = re.compile(r"^(?:[ACGTN]+|-)$")
_SAMPLE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


class MafError(ValueError):
    """The file cannot be read as a MAF at all (for example, a required column is missing)."""


@dataclass(frozen=True, slots=True)
class Mutation:
    sample: str
    gene: str
    vclass: str
    vtype: str
    ref: str
    alt: str
    protein: str = ""
    chrom: str = ""
    pos: int = 0
    t_ref: int | None = None
    t_alt: int | None = None
    context: str = ""

    @property
    def nonsynonymous(self) -> bool:
        return self.vclass in NONSYNONYMOUS

    @property
    def is_snv(self) -> bool:
        return self.vtype == "SNP" and len(self.ref) == 1 and len(self.alt) == 1 and self.ref != self.alt

    @property
    def vaf(self) -> float | None:
        if self.t_ref is None or self.t_alt is None or self.t_ref + self.t_alt == 0:
            return None
        return self.t_alt / (self.t_ref + self.t_alt)


@dataclass(frozen=True, slots=True)
class Filters:
    """Read-level filters. The defaults keep everything the source file reports."""

    min_vaf: float = 0.0
    min_alt_reads: int = 0


@dataclass(slots=True)
class QcReport:
    rows: int = 0
    kept: int = 0
    samples: int = 0
    rejected: Counter[str] = field(default_factory=Counter)
    examples: dict[str, int] = field(default_factory=dict)  # reason -> first line number
    warnings: list[str] = field(default_factory=list)

    def reject(self, reason: str, line: int) -> None:
        self.rejected[reason] += 1
        self.examples.setdefault(reason, line)

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "kept": self.kept,
            "samples": self.samples,
            "rejected": dict(sorted(self.rejected.items())),
            "first_line_per_reason": dict(sorted(self.examples.items())),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class MafData:
    mutations: list[Mutation]
    qc: QcReport

    def by_sample(self) -> dict[str, list[Mutation]]:
        out: dict[str, list[Mutation]] = {}
        for m in self.mutations:
            out.setdefault(m.sample, []).append(m)
        return out


def _cell(cells: list[str], index: dict[str, int], name: str) -> str:
    i = index.get(name)
    return cells[i].strip() if i is not None else ""


def _int(value: str) -> int | None:
    value = value.strip()
    if not value or value.upper() in {"NA", "NAN", "."}:
        return None
    try:
        number = int(float(value))
    except ValueError:
        return None
    return number if number >= 0 else None


def _lines(source: str | Path | Iterable[str]) -> Iterator[str]:
    if isinstance(source, Path):
        with source.open(encoding="utf-8", errors="replace") as handle:
            yield from handle
    elif isinstance(source, str):
        yield from io.StringIO(source)
    else:
        yield from source


def read_maf(source: str | Path | Iterable[str], filters: Filters | None = None, max_rows: int | None = None) -> MafData:
    """Parse MAF text, a path, or an iterable of lines.

    A string is treated as file content, not as a path; pass ``Path`` objects for files.
    ``max_rows`` bounds the work for untrusted uploads: reading stops with ``MafError`` beyond it.
    """
    filters = filters or Filters()
    qc = QcReport()
    header: list[str] | None = None
    index: dict[str, int] = {}
    seen: set[tuple] = set()
    mutations: list[Mutation] = []
    unknown_classes: Counter[str] = Counter()

    for number, raw in enumerate(_lines(source), start=1):
        line = raw.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        cells = line.split("\t")
        if header is None:
            header = [c.strip() for c in cells]
            index = {name: i for i, name in enumerate(header)}
            missing = [c for c in REQUIRED if c not in index]
            if missing:
                raise MafError(f"missing required column(s): {', '.join(missing)}")
            if "CONTEXT" not in index:
                qc.warnings.append("no CONTEXT column: 96-channel substitution profile is not available")
            if "t_alt_count" not in index or "t_ref_count" not in index:
                qc.warnings.append("no read counts: allele-frequency filters are not applied")
            continue

        qc.rows += 1
        if max_rows is not None and qc.rows > max_rows:
            raise MafError(f"more than {max_rows} mutation rows")
        if len(cells) < len(header):
            cells = cells + [""] * (len(header) - len(cells))

        get = partial(_cell, cells, index)

        sample = get("Tumor_Sample_Barcode")
        gene = get("Hugo_Symbol")
        vclass = get("Variant_Classification")
        vtype = get("Variant_Type").upper()
        ref = get("Reference_Allele").upper()
        alt = get("Tumor_Seq_Allele2").upper()

        if not sample:
            qc.reject("empty sample id", number)
            continue
        if not _SAMPLE.match(sample):
            qc.reject("sample id with unexpected characters (use pseudonymous ids)", number)
            continue
        if not gene:
            qc.reject("empty gene symbol", number)
            continue
        if vtype not in VARIANT_TYPES:
            qc.reject("unknown Variant_Type", number)
            continue
        if not _ALLELE.match(ref) or not _ALLELE.match(alt):
            qc.reject("invalid allele", number)
            continue
        if ref == alt:
            qc.reject("reference equals alternate allele", number)
            continue
        if vclass not in KNOWN_CLASSES:
            unknown_classes[vclass] += 1

        t_ref, t_alt = _int(get("t_ref_count")), _int(get("t_alt_count"))
        if t_alt is not None and t_alt < filters.min_alt_reads:
            qc.reject("too few alternate reads", number)
            continue
        depth = (t_ref or 0) + (t_alt or 0)
        if filters.min_vaf > 0 and t_ref is not None and t_alt is not None and depth > 0 and t_alt / depth < filters.min_vaf:
            qc.reject("allele frequency below threshold", number)
            continue

        chrom = get("Chromosome").removeprefix("chr")
        pos = _int(get("Start_Position")) or 0
        key = (sample, chrom, pos, ref, alt, gene) if pos else (sample, gene, vclass, ref, alt, get("HGVSp_Short"))
        if key in seen:
            qc.reject("duplicate row", number)
            continue
        seen.add(key)

        context = get("CONTEXT").upper()
        mutations.append(
            Mutation(
                sample=sample,
                gene=gene,
                vclass=vclass,
                vtype=vtype,
                ref=ref,
                alt=alt,
                protein=get("HGVSp_Short"),
                chrom=chrom,
                pos=pos,
                t_ref=t_ref,
                t_alt=t_alt,
                context=context if len(context) % 2 == 1 and set(context) <= set("ACGTN") else "",
            )
        )

    if header is None:
        raise MafError("no header row found")
    for vclass, count in unknown_classes.most_common():
        qc.warnings.append(f"unrecognised Variant_Classification '{vclass}' on {count} row(s); counted as not protein-changing")
    qc.kept = len(mutations)
    qc.samples = len({m.sample for m in mutations})
    return MafData(mutations=mutations, qc=qc)


def patient_id(sample: str) -> str:
    """TCGA sample barcodes start with the 12-character patient barcode; other ids are used as given."""
    if sample.startswith("TCGA-") and len(sample) >= 12:
        return sample[:12]
    return sample
