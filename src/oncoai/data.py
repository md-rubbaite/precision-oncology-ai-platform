"""Public study data: the pinned manifest, verified download, and loading a study into labelled profiles."""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import tomllib
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .features import DEFAULT_EXOME_MB, Profile, profiles
from .maf import Filters, patient_id, read_maf

# The ten studies used for the published benchmark (docs/data/validation.md explains why ten).
BENCHMARK = ("acc", "dlbc", "gbm", "kich", "lgg", "meso", "ov", "tgct", "ucs", "uvm")


@dataclass(frozen=True, slots=True)
class FileSpec:
    file: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class Study:
    code: str
    folder: str
    mutations: FileSpec
    clinical: FileSpec


@dataclass(frozen=True, slots=True)
class Manifest:
    repository: str
    commit: str
    url: str
    studies: dict[str, Study]

    def url_for(self, study: Study, spec: FileSpec) -> str:
        return self.url.format(commit=self.commit, study=study.folder, file=spec.file)


def load_manifest(path: Path | None = None) -> Manifest:
    text = path.read_text(encoding="utf-8") if path else resources.files("oncoai").joinpath("studies.toml").read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    studies = {}
    for code, item in raw["studies"].items():
        studies[code] = Study(
            code=code,
            folder=item["folder"],
            mutations=FileSpec(**item["mutations"]),
            clinical=FileSpec(**item["clinical"]),
        )
    src = raw["source"]
    return Manifest(repository=src["repository"], commit=src["commit"], url=src["url"], studies=studies)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class DownloadError(RuntimeError):
    pass


Opener = Callable[[str], object]


def _default_open(url: str):
    if not url.startswith("https://"):
        raise DownloadError(f"refusing non-HTTPS URL: {url}")
    return urllib.request.urlopen(url, timeout=120)  # noqa: S310 - scheme checked above


def fetch_file(url: str, spec: FileSpec, dest: Path, opener: Opener | None = None) -> str:
    """Download ``url`` to ``dest`` and check size and SHA-256. Returns 'cached' or 'downloaded'.

    The file is written to a temporary name and only moved into place after both checks pass,
    so a failed or tampered download never leaves a file that looks valid.
    """
    if dest.exists() and dest.stat().st_size == spec.size and sha256_file(dest) == spec.sha256:
        return "cached"
    opener = opener or _default_open
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".part-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, opener(url) as response:  # type: ignore[attr-defined]
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                size += len(block)
                if size > spec.size:
                    raise DownloadError(f"{dest.name}: larger than the expected {spec.size} bytes")
                digest.update(block)
                out.write(block)
        if size != spec.size:
            raise DownloadError(f"{dest.name}: got {size} bytes, expected {spec.size}")
        if digest.hexdigest() != spec.sha256:
            raise DownloadError(f"{dest.name}: SHA-256 does not match the manifest")
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return "downloaded"


def fetch(codes: Iterable[str], dest: Path, manifest: Manifest | None = None, opener: Opener | None = None) -> dict[str, str]:
    manifest = manifest or load_manifest()
    results: dict[str, str] = {}
    for code in codes:
        if code not in manifest.studies:
            raise KeyError(f"unknown study '{code}'")
        study = manifest.studies[code]
        for kind, spec in (("mutations", study.mutations), ("clinical", study.clinical)):
            target = dest / code / spec.file
            try:
                results[f"{code}/{kind}"] = fetch_file(manifest.url_for(study, spec), spec, target, opener)
            except (DownloadError, OSError) as exc:
                results[f"{code}/{kind}"] = f"failed: {exc}"
    return results


@dataclass(slots=True)
class Cohort:
    profiles: list[Profile]
    labels: list[str]
    patients: list[str]
    published_tmb: dict[str, float]
    qc: dict[str, dict]

    def __len__(self) -> int:
        return len(self.profiles)

    def extend(self, other: Cohort) -> None:
        self.profiles += other.profiles
        self.labels += other.labels
        self.patients += other.patients
        self.published_tmb.update(other.published_tmb)
        self.qc.update(other.qc)


def read_clinical(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        rows = [line.rstrip("\r\n").split("\t") for line in handle if line.strip() and not line.startswith("#")]
    if not rows:
        return {}
    header = rows[0]
    if "SAMPLE_ID" not in header:
        raise ValueError(f"{path}: no SAMPLE_ID column")
    out = {}
    for row in rows[1:]:
        record = dict(zip(header, row + [""] * (len(header) - len(row)), strict=True))
        out[record["SAMPLE_ID"]] = record
    return out


def load_study(root: Path, code: str, filters: Filters | None = None, exome_mb: float = DEFAULT_EXOME_MB) -> Cohort:
    """Load one fetched study. Label = study code in upper case; only samples with mutations are used."""
    folder = root / code
    data = read_maf(folder / "data_mutations.txt", filters)
    clinical_path = folder / "data_clinical_sample.txt"
    clinical = read_clinical(clinical_path) if clinical_path.exists() else {}
    profs = profiles(data.by_sample(), exome_mb)
    published: dict[str, float] = {}
    patients: list[str] = []
    for sample in profs:
        record = clinical.get(sample, {})
        patients.append(record.get("PATIENT_ID") or patient_id(sample))
        with contextlib.suppress(ValueError):
            published[sample] = float(record.get("TMB_NONSYNONYMOUS", ""))
    label = code.upper()
    return Cohort(
        profiles=list(profs.values()),
        labels=[label] * len(profs),
        patients=patients,
        published_tmb=published,
        qc={label: data.qc.as_dict()},
    )


def load_cohort(root: Path, codes: Iterable[str], filters: Filters | None = None) -> Cohort:
    cohort = Cohort([], [], [], {}, {})
    for code in codes:
        cohort.extend(load_study(root, code, filters))
    return cohort
