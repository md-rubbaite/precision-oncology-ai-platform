import json

import numpy as np
import pytest

from oncoai.model import FORMAT, RESEARCH_USE, Model, TrainParams, fit, softmax


def test_numpy_scoring_matches_scikit_learn(synthetic_profiles):
    from sklearn.linear_model import LogisticRegression

    items, labels = synthetic_profiles
    model = fit(items, labels, TrainParams(C=0.5))
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000).fit(model.space.transform(items), labels)
    assert np.allclose(model.predict_proba(items), clf.predict_proba(model.space.transform(items)), atol=1e-8)


def test_two_class_model_expands_to_softmax(synthetic_profiles):
    from sklearn.linear_model import LogisticRegression

    items, labels = synthetic_profiles
    keep = [i for i, y in enumerate(labels) if y in ("SYNA", "SYND")]
    sub, ys = [items[i] for i in keep], [labels[i] for i in keep]
    model = fit(sub, ys)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(model.space.transform(sub), ys)
    assert model.coef.shape[0] == 2
    assert np.allclose(model.predict_proba(sub), clf.predict_proba(model.space.transform(sub)), atol=1e-8)


def test_json_round_trip_keeps_predictions(trained, synthetic_profiles, tmp_path):
    model, _ = trained
    items, _ = synthetic_profiles
    path = tmp_path / "m.json"
    model.save(path)
    data = json.loads(path.read_text())
    assert data["format"] == FORMAT and data["notice"] == RESEARCH_USE
    again = Model.load(path)
    assert np.allclose(again.predict_proba(items), model.predict_proba(items), atol=1e-6)
    assert again.threshold == model.threshold and again.classes == model.classes


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"format": "pickle"}, "not an"),
        ({"classes": ["A"]}, "two distinct"),
        ({"coef": [[0.0]]}, "shape"),
        ({"threshold": 1.5}, "between 0 and 1"),
    ],
)
def test_model_file_is_validated(trained, change, message):
    model, _ = trained
    with pytest.raises(ValueError, match=message):
        Model.from_dict(model.to_dict() | change)


def test_non_finite_coefficients_are_refused(trained):
    model, _ = trained
    data = model.to_dict()
    data["coef"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        Model.from_dict(data)


def test_oversized_model_file_is_refused(tmp_path, monkeypatch):
    import oncoai.model as m

    path = tmp_path / "big.json"
    path.write_text("x" * 100)
    monkeypatch.setattr(m, "MAX_MODEL_BYTES", 10)
    with pytest.raises(ValueError, match="larger"):
        Model.load(path)


def test_predictions_abstain_below_threshold(trained, synthetic_profiles):
    model, _ = trained
    items, _ = synthetic_profiles
    model_hi = Model.from_dict(model.to_dict() | {"threshold": 1.0})
    preds = model_hi.predict(items[:5])
    assert all(p.abstained for p in preds)
    d = preds[0].as_dict()
    assert d["call"] is None and d["most_likely"] in model.classes and d["notice"] == RESEARCH_USE
    assert sum(x["p"] for x in d["probabilities"]) == pytest.approx(1.0, abs=1e-3)


def test_explanations_are_positive_and_named(trained, synthetic_profiles):
    model, _ = trained
    items, _ = synthetic_profiles
    pred = model.predict(items[:1])[0]
    assert pred.reasons and all(r["contribution"] > 0 for r in pred.reasons)
    assert all(r["feature"] in model.space.names for r in pred.reasons)


def test_softmax_is_stable():
    p = softmax(np.array([[1000.0, 1001.0], [0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0) and np.isfinite(p).all()


def test_params_dict_round_trip():
    assert TrainParams(**TrainParams(C=0.3).to_dict()) == TrainParams(C=0.3)
