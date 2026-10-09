import math

import numpy as np
import pytest

from data_layer import MOODS
from model_service import ModelNotTrainedError, MoodModel, validate_features
from train_model import save_model, train_and_evaluate

POINT = {"valence": 0.8, "energy": 0.8, "danceability": 0.5, "tempo": 120.0}


@pytest.fixture
def trained(catalog_df):
    return train_and_evaluate(catalog_df, k=5)


def test_split_and_metrics(trained, catalog_df):
    bundle, metrics = trained
    assert metrics["n_train"] + metrics["n_test"] == len(catalog_df)
    assert metrics["n_test"] == math.ceil(0.2 * len(catalog_df))
    # The scaler must have been fitted on the training rows only.
    assert bundle["pipeline"].named_steps["scaler"].n_samples_seen_ == metrics["n_train"]
    assert metrics["test_accuracy"] > metrics["majority_baseline"]["accuracy"]
    assert np.array(metrics["confusion_matrix"]["matrix"]).sum() == metrics["n_test"]


def test_training_is_reproducible(catalog_df):
    _, a = train_and_evaluate(catalog_df, k=5)
    _, b = train_and_evaluate(catalog_df, k=5)
    assert a["test_accuracy"] == b["test_accuracy"]


@pytest.mark.parametrize("k", [0, 10_000])
def test_invalid_k_rejected(catalog_df, k):
    with pytest.raises(ValueError, match="k="):
        train_and_evaluate(catalog_df, k=k)


def test_save_load_predict_explain(tmp_path, trained, catalog_df):
    bundle, metrics = trained
    catalog_path = tmp_path / "songs.csv"
    catalog_df.to_csv(catalog_path, index=False)
    model_path, meta_path = tmp_path / "m.joblib", tmp_path / "m.json"
    save_model(bundle, metrics, catalog_path=catalog_path, model_path=model_path, meta_path=meta_path)

    model = MoodModel.load(model_path, meta_path)
    assert model.meta["metrics"]["k"] == 5

    result = model.predict(POINT)
    assert result["mood"] in MOODS
    assert set(result["probabilities"]) == set(MOODS)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert result["rule_mood"] == "Happy / Excited"

    exp = model.explain(POINT)
    assert sum(exp["votes"].values()) == model.k == len(exp["neighbours"])
    distances = [n["distance"] for n in exp["neighbours"]]
    assert distances == sorted(distances)
    # The winning vote matches the prediction (uniform-weight KNN).
    assert max(exp["votes"], key=exp["votes"].get) == result["mood"]


def test_explain_can_exclude_the_song_itself(trained, catalog_df):
    model = MoodModel(trained[0])
    own_id = str(model.train_track_ids[0])
    row = catalog_df.set_index("track_id").loc[own_id]
    features = {f: float(row[f]) for f in POINT}
    assert model.explain(features)["neighbours"][0]["track_id"] == own_id
    excluded = model.explain(features, exclude_track_id=own_id)["neighbours"]
    assert own_id not in {n["track_id"] for n in excluded} and len(excluded) == model.k


def test_missing_model_file(tmp_path):
    with pytest.raises(ModelNotTrainedError, match="python -m train_model"):
        MoodModel.load(tmp_path / "missing.joblib", tmp_path / "missing.json")


def test_feature_schema_mismatch(trained):
    bundle = {**trained[0], "features": ["energy", "valence", "danceability", "tempo"]}
    with pytest.raises(ValueError, match="Retrain"):
        MoodModel(bundle)


@pytest.mark.parametrize("bad, message", [
    ({"valence": 0.5, "energy": 0.5, "danceability": 0.5}, "Missing"),
    ({**POINT, "valence": 1.2}, "between 0 and 1"),
    ({**POINT, "energy": "loud"}, "must be a number"),
    ({**POINT, "tempo": 0}, "positive"),
    ({**POINT, "danceability": float("nan")}, "finite"),
])
def test_validate_features_rejects_bad_input(bad, message):
    with pytest.raises(ValueError, match=message):
        validate_features(bad)
