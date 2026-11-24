import json

import pytest
from fastapi.testclient import TestClient

from oncoai.api import MAX_BODY_BYTES, MAX_SAMPLES, KeyStore, create_app
from oncoai.model import RESEARCH_USE
from oncoai.synthetic import maf_text

VIEWER = "viewer-key-0123456789abcdefgh"
ANALYST = "analyst-key-0123456789abcdefg"
ADMIN = "admin-key-0123456789abcdefghij"
KEYS = f"viewer:{VIEWER},analyst:{ANALYST},admin:{ADMIN}"
TSV = {"content-type": "text/tab-separated-values"}


@pytest.fixture()
def client(trained, tmp_path):
    model, _ = trained
    path = tmp_path / "model.json"
    model.save(path)
    app = create_app(model, KeyStore(KEYS), model_path=path, example=maf_text(per_type=1, seed=8))
    with TestClient(app) as c:
        yield c


def h(key, extra=None):
    return {"x-api-key": key, **(extra or {})}


def test_keystore_rules():
    with pytest.raises(ValueError, match="too short"):
        KeyStore("viewer:short")
    with pytest.raises(ValueError, match="invalid key entry"):
        KeyStore("root:" + "x" * 30)
    with pytest.raises(ValueError, match="no API keys"):
        KeyStore("")
    ks = KeyStore(KEYS)
    assert ks.role(ADMIN) == "admin" and ks.role("nope") is None and ks.role(None) is None


def test_health_ready_and_root(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"ready": True}
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/dashboard"


def test_authentication_and_roles(client):
    assert client.get("/v1/model").status_code == 401
    assert client.get("/v1/model", headers=h("wrong" * 6)).status_code == 401
    assert client.get("/v1/whoami", headers=h(VIEWER)).json() == {"role": "viewer"}
    r = client.post("/v1/analyse", headers=h(VIEWER, TSV), content="x")
    assert r.status_code == 403 and "analyst" in r.json()["detail"]
    assert client.post("/v1/admin/reload", headers=h(ANALYST)).status_code == 403


def test_model_card_and_cohort(client, trained):
    model, _ = trained
    card = client.get("/v1/model", headers=h(VIEWER)).json()
    assert card["classes"] == model.classes and card["notice"] == RESEARCH_USE
    assert card["features"]["total"] == len(model.space.names) and "accuracy" in card["cross_validation"]
    cohort = client.get("/v1/cohort", headers=h(VIEWER)).json()
    assert set(cohort["labels"]) == set(model.classes)


def test_analyse_returns_profiles_predictions_and_notice(client):
    text = maf_text(per_type=1, seed=8)
    r = client.post("/v1/analyse", headers=h(ANALYST, TSV), content=text)
    assert r.status_code == 200
    body = r.json()
    assert body["notice"] == RESEARCH_USE and body["qc"]["samples"] == 5 and len(body["samples"]) == 5
    s = body["samples"][0]
    assert s["prediction"]["notice"] == RESEARCH_USE and s["profile"]["tmb_per_mb"] >= 0
    assert set(body["cohort_tmb"]) == set(client.get("/v1/model", headers=h(VIEWER)).json()["classes"])
    assert all(g.startswith("SG") for g in s["model_genes_hit"])


def test_analyse_applies_read_filters(client):
    text = maf_text(per_type=1, seed=8)
    loose = client.post("/v1/analyse", headers=h(ANALYST, TSV), content=text).json()["qc"]["kept"]
    strict = client.post("/v1/analyse?min_alt_reads=40", headers=h(ANALYST, TSV), content=text).json()["qc"]["kept"]
    assert strict < loose
    assert client.post("/v1/analyse?min_vaf=2", headers=h(ANALYST, TSV), content=text).status_code == 422
    assert client.post("/v1/analyse?min_vaf=abc", headers=h(ANALYST, TSV), content=text).status_code == 422


def test_analyse_input_limits(client):
    assert client.post("/v1/analyse", headers=h(ANALYST, {"content-type": "application/json"}), content="{}").status_code == 415
    assert client.post("/v1/analyse", headers=h(ANALYST, TSV), content="no header here").status_code == 422
    assert client.post("/v1/analyse", headers=h(ANALYST, TSV), content=b"\xff\xfe\x00bad").status_code == 422
    big = "x" * (MAX_BODY_BYTES + 10)
    assert client.post("/v1/analyse", headers=h(ANALYST, TSV), content=big).status_code == 413
    many = maf_text(per_type=MAX_SAMPLES // 5 + 1, seed=2)
    r = client.post("/v1/analyse", headers=h(ANALYST, TSV), content=many)
    assert r.status_code == 422 and "at most" in r.json()["detail"]


def test_audit_records_hash_not_content(client):
    text = maf_text(per_type=1, seed=8)
    client.post("/v1/analyse", headers=h(ANALYST, TSV), content=text)
    client.post("/v1/analyse", headers=h(ANALYST, TSV), content="bad")
    entries = client.get("/v1/admin/audit", headers=h(ADMIN)).json()["entries"]
    assert entries[0]["outcome"].startswith("rejected") and entries[1]["outcome"] == "ok"
    assert entries[1]["samples"] == 5 and len(entries[1]["sha256"]) == 64
    dumped = json.dumps(entries)
    assert "SYN-" not in dumped and "Missense" not in dumped


def test_admin_reload(client, tmp_path):
    r = client.post("/v1/admin/reload", headers=h(ADMIN))
    assert r.status_code == 200 and r.json()["reloaded"] is True


def test_reload_refused_for_demo_model(trained):
    model, _ = trained
    with TestClient(create_app(model, KeyStore(KEYS))) as c:
        assert c.post("/v1/admin/reload", headers=h(ADMIN)).status_code == 409
        assert c.get("/v1/example", headers=h(ANALYST)).status_code == 404


def test_reload_reports_broken_model_file(client, tmp_path):
    state = client.app.state.oncoai
    state.model_path.write_text("{}")
    r = client.post("/v1/admin/reload", headers=h(ADMIN))
    assert r.status_code == 422 and state.model is not None


def test_example_metrics_dashboard_and_headers(client):
    ex = client.get("/v1/example", headers=h(ANALYST))
    assert ex.status_code == 200 and ex.text.startswith("#synthetic")
    client.post("/v1/analyse", headers=h(ANALYST, TSV), content=ex.text)
    m = client.get("/metrics").text
    assert "oncoai_samples_analysed_total 5" in m and 'oncoai_http_requests_total{method="POST",status="200"}' in m
    d = client.get("/dashboard")
    assert d.status_code == 200 and "Research use only" in d.text
    assert "frame-ancestors 'none'" in d.headers["content-security-policy"]
    r = client.get("/healthz", headers={"x-request-id": "abc"})
    assert (
        r.headers["x-request-id"] == "abc" and r.headers["cache-control"] == "no-store" and r.headers["x-content-type-options"] == "nosniff"
    )


def test_dashboard_does_not_use_innerhtml_with_data():
    from importlib import resources

    html = resources.files("oncoai").joinpath("dashboard/index.html").read_text()
    assert "innerHTML=" not in html.replace('tb.innerHTML=\'<tr><td colspan="6" class="muted">No usable mutations.</td></tr>\'', "")
