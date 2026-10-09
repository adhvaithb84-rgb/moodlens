"""The DEAM-trained upload model, exercised with a small fake bundle (no CLAP download)."""
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from audio_embeddings import SR_FEATURES, librosa_feature_names, middle_segment
from audio_mood import AudioModelNotTrainedError, AudioMoodModel
from train_audio_mood import quadrant_scores
from tests.test_audio_features import tone, to_wav


def fake_bundle(valence=0.8, energy=0.3, feature_set="librosa"):
    n = len(librosa_feature_names())
    models = {}
    for target, value in {"valence": valence, "energy": energy}.items():
        models[target] = DummyRegressor(strategy="constant", constant=value).fit(np.zeros((2, n)), [value] * 2)
    return {"feature_set": feature_set, "models": models, "targets": ["valence", "energy"],
            "librosa_features": librosa_feature_names()}


def test_predict_from_audio_bytes():
    model = AudioMoodModel(fake_bundle(), {"results": {"librosa": {"mood_accuracy": 0.5}}, "n_songs": 10})
    out = model.predict(to_wav(tone([440, 554, 659], seconds=8)))
    assert out == {"valence": pytest.approx(0.8), "energy": pytest.approx(0.3)}
    assert not model.uses_clap
    assert model.quality()["chosen"]["mood_accuracy"] == 0.5


def test_predictions_are_clipped():
    out = AudioMoodModel(fake_bundle(valence=1.7, energy=-0.4)).predict(to_wav(tone([440], seconds=8)))
    assert out == {"valence": 1.0, "energy": 0.0}


def test_schema_change_detected():
    bundle = fake_bundle()
    bundle["librosa_features"] = bundle["librosa_features"][:-1]
    with pytest.raises(ValueError, match="Retrain"):
        AudioMoodModel(bundle)


def test_missing_model(tmp_path):
    with pytest.raises(AudioModelNotTrainedError, match="train_audio_mood"):
        AudioMoodModel.load(tmp_path / "none.joblib", tmp_path / "none.json")


def test_falls_back_to_lite_model_without_torch(tmp_path, monkeypatch):
    import joblib

    import audio_mood

    full, lite = tmp_path / "full.joblib", tmp_path / "lite.joblib"
    joblib.dump(fake_bundle(feature_set="both"), full)
    joblib.dump(fake_bundle(feature_set="librosa"), lite)
    monkeypatch.setattr(audio_mood, "MODEL_PATH", full)
    monkeypatch.setattr(audio_mood, "LITE_MODEL_PATH", lite)
    monkeypatch.setattr(audio_mood, "META_PATH", tmp_path / "meta.json")

    monkeypatch.setattr(audio_mood, "clap_available", lambda: True)
    assert AudioMoodModel.load(meta_path=tmp_path / "meta.json").feature_set == "both"
    monkeypatch.setattr(audio_mood, "clap_available", lambda: False)
    assert AudioMoodModel.load(meta_path=tmp_path / "meta.json").feature_set == "librosa"
    with pytest.raises(AudioModelNotTrainedError, match="torch"):
        AudioMoodModel.load(full, tmp_path / "meta.json")


def test_middle_segment():
    y = np.arange(100 * SR_FEATURES)
    seg = middle_segment(y, SR_FEATURES, seconds=10)
    assert len(seg) == 10 * SR_FEATURES and seg[0] == 45 * SR_FEATURES
    assert len(middle_segment(y[:5], SR_FEATURES, seconds=10)) == 5


def test_quadrant_scores():
    s = quadrant_scores([0.9, 0.1, 0.9, 0.1], [0.9, 0.9, 0.1, 0.1], [0.8, 0.2, 0.2, 0.2], [0.8, 0.8, 0.2, 0.6])
    assert s["mood_accuracy"] == 0.5
    assert s["valence_side_accuracy"] == 0.75 and s["energy_side_accuracy"] == 0.75
    assert s["majority_baseline"] == 0.25
