import json
from pathlib import Path

import pytest

import oncoai.data as data_mod
from oncoai.cli import build_parser, main
from oncoai.synthetic import generate_rows, maf_text, to_maf


@pytest.fixture()
def maf_file(tmp_path: Path) -> Path:
    p = tmp_path / "cohort.maf"
    p.write_text(maf_text(per_type=2, seed=4), encoding="utf-8")
    return p


def test_qc_text_and_json(maf_file, capsys):
    assert main(["qc", str(maf_file)]) == 0
    assert "samples 10" in capsys.readouterr().out
    assert main(["qc", str(maf_file), "--json", "--min-alt-reads", "1000"]) == 1
    assert json.loads(capsys.readouterr().out)["kept"] == 0


def test_qc_reports_unreadable_file(tmp_path, capsys):
    bad = tmp_path / "bad.maf"
    bad.write_text("not\ta\tmaf\n")
    assert main(["qc", str(bad)]) == 2
    assert "missing required" in capsys.readouterr().err


def test_profile(maf_file, capsys):
    assert main(["profile", str(maf_file)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out) == 10 and {"tmb_per_mb", "sbs6"} <= set(out[0])


def test_demo_runs_end_to_end(tmp_path, capsys):
    out = tmp_path / "demo.json"
    assert main(["demo", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "synthetic data only" in text and "rejected" in text and "Done." in text
    assert json.loads(out.read_text())["format"] == "oncoai-model/1"


def _studies(root: Path) -> None:
    rows = generate_rows(per_type=12, seed=6)
    for letter in "ABCDE":
        folder = root / f"syn{letter.lower()}"
        folder.mkdir(parents=True)
        (folder / "data_mutations.txt").write_text(to_maf([r for r in rows if r["Tumor_Sample_Barcode"].startswith(f"SYN-{letter}")]))


def test_train_predict_and_cohort(tmp_path, maf_file, capsys):
    _studies(tmp_path / "data")
    model, report = tmp_path / "m.json", tmp_path / "r.json"
    args = [
        "train",
        "--data",
        str(tmp_path / "data"),
        "--studies",
        "syna,synb,sync,synd,syne",
        "--out",
        str(model),
        "--report",
        str(report),
        "--folds",
        "3",
    ]
    assert main(args) == 0
    assert "cross-validated" in capsys.readouterr().out
    assert json.loads(report.read_text())["samples"] == 60
    assert main(["predict", str(maf_file), "--model", str(model)]) == 0
    preds = json.loads(capsys.readouterr().out)["predictions"]
    assert len(preds) == 10 and all(p["notice"].startswith("Research use only") for p in preds)
    assert main(["cohort", "syna", "--data", str(tmp_path / "data"), "--pairs", "3"]) == 0
    assert "SYNA: 12 samples" in capsys.readouterr().out


def test_fetch_uses_manifest_and_reports_failures(tmp_path, monkeypatch, capsys):
    import io

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    monkeypatch.setattr(data_mod, "_default_open", lambda url: Resp(b"not the real file"))
    assert main(["fetch", "--studies", "uvm", "--dest", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "uvm/mutations" in out and "failed" in out and "ODbL" in out


def test_study_selection():
    from oncoai.cli import _studies as pick

    assert pick("benchmark")[0] == "acc" and len(pick("all")) == 32 and pick("GBM, lgg") == ["gbm", "lgg"]


def test_serve_requires_keys(monkeypatch, capsys):
    monkeypatch.delenv("ONCOAI_API_KEYS", raising=False)
    assert main(["serve"]) == 2
    assert "ONCOAI_API_KEYS" in capsys.readouterr().err


def test_parser_has_all_commands():
    sub = next(a for a in build_parser()._actions if a.dest == "command")
    assert set(sub.choices) == {"fetch", "qc", "profile", "cohort", "train", "predict", "demo", "serve"}
