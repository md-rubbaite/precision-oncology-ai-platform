import numpy as np
import pytest

import oncoai.evaluate as ev
from oncoai.evaluate import abstention_table, choose_threshold, cross_validate, expected_calibration_error, grouped_folds, metrics


def test_metrics_on_a_small_known_case():
    classes = ["A", "B", "C"]
    y = ["A", "A", "B", "C"]
    proba = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1], [0.5, 0.1, 0.4]])
    m = metrics(y, proba, classes)
    assert m["accuracy"] == 0.5
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 1 + 0) / 3, abs=1e-4)
    assert m["top3_accuracy"] == 1.0
    assert m["log_loss"] == pytest.approx(-np.mean(np.log([0.8, 0.3, 0.7, 0.4])), abs=1e-4)
    assert m["per_class"]["B"] == {"precision": 0.5, "recall": 1.0, "f1": 0.667, "support": 1}
    assert m["confusion"]["matrix"] == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]


def test_metrics_agree_with_scikit_learn(synthetic_profiles):
    from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

    rng = np.random.default_rng(1)
    classes = ["A", "B", "C", "D"]
    y = list(rng.choice(classes, 200))
    proba = rng.dirichlet(np.ones(4), 200)
    m = metrics(y, proba, classes)
    pred = [classes[i] for i in proba.argmax(1)]
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred), abs=1e-4)
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"), abs=1e-4)
    assert m["log_loss"] == pytest.approx(log_loss(y, proba, labels=classes), abs=1e-4)


def test_ece_is_zero_when_confidence_matches_accuracy():
    proba = np.array([[1.0, 0.0]] * 10)
    assert expected_calibration_error(proba, np.zeros(10, dtype=int)) == 0.0
    assert expected_calibration_error(proba, np.ones(10, dtype=int)) == pytest.approx(1.0)


def test_abstention_table_and_threshold_choice():
    classes = ["A", "B"]
    proba = np.array([[0.95, 0.05]] * 20 + [[0.55, 0.45]] * 20)
    y = ["A"] * 20 + ["A"] * 10 + ["B"] * 10
    rows = {r["threshold"]: r for r in abstention_table(y, proba, classes)}
    assert rows[0.0]["accuracy_when_answered"] == 0.75 and rows[0.9]["coverage"] == 0.5 and rows[0.9]["accuracy_when_answered"] == 1.0
    assert choose_threshold(y, proba, classes, 0.95) == 0.6
    assert choose_threshold(y, proba, classes, 0.7) == 0.0
    assert choose_threshold(["B"] * 40, proba, classes, 0.5) == 1.0


def test_grouped_folds_never_split_a_patient():
    labels = ["A"] * 30 + ["B"] * 30
    groups = [f"P{i // 2}" for i in range(60)]  # two samples per patient
    folds = grouped_folds(labels, groups, 5, seed=1)
    assert len(folds) == 5
    for tr, te in folds:
        assert not ({groups[i] for i in tr} & {groups[i] for i in te})
        assert {labels[i] for i in te} == {"A", "B"}
    assert sorted(np.concatenate([te for _, te in folds]).tolist()) == list(range(60))


def test_folds_shrink_for_small_classes():
    assert len(grouped_folds(["A"] * 10 + ["B"] * 3, [str(i) for i in range(13)], 5, 0)) == 3


def test_cross_validation_never_trains_on_held_out_patients(synthetic_profiles, monkeypatch):
    items, labels = synthetic_profiles
    groups = [p.sample for p in items]
    seen: list[set[str]] = []
    real_fit = ev.fit

    def spy(train_items, train_labels, params):
        seen.append({p.sample for p in train_items})
        return real_fit(train_items, train_labels, params)

    monkeypatch.setattr(ev, "fit", spy)
    result = cross_validate(items, labels, groups, n_splits=3, seed=2, tune_c=False, with_boosting=False)
    folds = grouped_folds(labels, groups, 3, 2)
    for f, (_, te) in enumerate(folds):
        held_out = {items[i].sample for i in te}
        assert not (seen[2 * f] & held_out) and not (seen[2 * f + 1] & held_out)
    assert result.proba["logistic"].shape == (len(items), 5)
    assert np.allclose(result.proba["logistic"].sum(axis=1), 1.0)


def test_cross_validation_report_has_baselines(synthetic_profiles):
    items, labels = synthetic_profiles
    result = cross_validate(items, labels, [p.sample for p in items], n_splits=3, seed=2, tune_c=True, with_boosting=True)
    s = result.summary()
    assert set(s) == {"logistic", "mutational_process_only", "class_prior", "gradient_boosting"}
    assert s["logistic"]["pooled"]["accuracy"] > s["class_prior"]["pooled"]["accuracy"]
    assert all(c in ev.C_GRID for c in result.chosen_c)
    assert "sd" in s["logistic"]["across_folds"]["accuracy"]


def test_training_report_and_model_meta(trained):
    model, report = trained
    assert report["abstention"]["threshold"] == model.threshold
    assert report["cross_validation"]["chosen_c_per_fold"]
    assert set(model.meta["cohort"]) == set(model.classes)
    assert model.meta["cv"]["accuracy"] == report["cross_validation"]["methods"]["logistic"]["pooled"]["accuracy"]
