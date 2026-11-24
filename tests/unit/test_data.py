import hashlib
import io
import re
from pathlib import Path

import pytest

from oncoai.data import (
    BENCHMARK,
    DownloadError,
    FileSpec,
    fetch,
    fetch_file,
    load_cohort,
    load_manifest,
    load_study,
    read_clinical,
    sha256_file,
)
from oncoai.synthetic import generate_rows, to_maf


def test_manifest_pins_all_32_studies():
    m = load_manifest()
    assert m.repository == "cBioPortal/datahub" and re.fullmatch(r"[0-9a-f]{40}", m.commit)
    assert len(m.studies) == 32 and set(BENCHMARK) <= set(m.studies)
    for s in m.studies.values():
        assert re.fullmatch(r"[0-9a-f]{64}", s.mutations.sha256) and s.mutations.size > 0
        assert m.url_for(s, s.mutations).startswith("https://media.githubusercontent.com/media/cBioPortal/datahub/" + m.commit)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def opener_for(payload: bytes):
    calls = []

    def opener(url):
        calls.append(url)
        return FakeResponse(payload)

    return opener, calls


def spec_for(payload: bytes) -> FileSpec:
    return FileSpec("f.txt", hashlib.sha256(payload).hexdigest(), len(payload))


def test_fetch_file_verifies_and_caches(tmp_path: Path):
    payload = b"hello world\n" * 100
    opener, calls = opener_for(payload)
    dest = tmp_path / "s" / "f.txt"
    assert fetch_file("https://x/f", spec_for(payload), dest, opener) == "downloaded"
    assert dest.read_bytes() == payload and sha256_file(dest) == spec_for(payload).sha256
    assert fetch_file("https://x/f", spec_for(payload), dest, opener) == "cached"
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("served", "match"),
    [(b"tampered!!", "SHA-256"), (b"short", "expected"), (b"x" * 50, "larger")],
)
def test_fetch_file_rejects_bad_downloads_and_leaves_nothing(tmp_path: Path, served, match):
    expected = b"0123456789"
    opener, _ = opener_for(served)
    dest = tmp_path / "f.txt"
    with pytest.raises(DownloadError, match=match):
        fetch_file("https://x/f", spec_for(expected), dest, opener)
    assert list(tmp_path.iterdir()) == []


def test_default_opener_refuses_plain_http():
    from oncoai.data import _default_open

    with pytest.raises(DownloadError, match="non-HTTPS"):
        _default_open("http://example.org/x")


def test_fetch_reports_failures_per_file(tmp_path: Path):
    opener, _ = opener_for(b"nope")
    results = fetch(["uvm"], tmp_path, opener=opener)
    assert set(results) == {"uvm/mutations", "uvm/clinical"}
    assert all(v.startswith("failed") for v in results.values())
    with pytest.raises(KeyError):
        fetch(["nosuchstudy"], tmp_path, opener=opener)


def _write_study(root: Path, code: str, letter: str) -> None:
    rows = [r for r in generate_rows(per_type=6, seed=2) if r["Tumor_Sample_Barcode"].startswith(f"SYN-{letter}")]
    folder = root / code
    folder.mkdir(parents=True)
    (folder / "data_mutations.txt").write_text(to_maf(rows), encoding="utf-8")
    samples = sorted({r["Tumor_Sample_Barcode"] for r in rows})
    lines = ["#Patient\tSample\tTMB", "#x\tx\tx", "PATIENT_ID\tSAMPLE_ID\tTMB_NONSYNONYMOUS"]
    lines += [f"{s[:-3]}\t{s}\t1.5" for s in samples]
    (folder / "data_clinical_sample.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_study_and_cohort(tmp_path: Path):
    _write_study(tmp_path, "syna", "A")
    _write_study(tmp_path, "synd", "D")
    study = load_study(tmp_path, "syna")
    assert len(study) == 6 and set(study.labels) == {"SYNA"}
    assert study.patients[0] == study.profiles[0].sample[:-3]
    assert all(v == 1.5 for v in study.published_tmb.values())
    cohort = load_cohort(tmp_path, ["syna", "synd"])
    assert len(cohort) == 12 and set(cohort.labels) == {"SYNA", "SYND"} and set(cohort.qc) == {"SYNA", "SYND"}


def test_read_clinical_requires_sample_id(tmp_path: Path):
    p = tmp_path / "c.txt"
    p.write_text("A\tB\n1\t2\n")
    with pytest.raises(ValueError, match="SAMPLE_ID"):
        read_clinical(p)
    p.write_text("#only comments\n")
    assert read_clinical(p) == {}
