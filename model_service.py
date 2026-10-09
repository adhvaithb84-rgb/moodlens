"""Load the trained MoodLens model and serve predictions with explanations."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_layer import FEATURES, MOODS, THRESHOLD, UNIT_FEATURES, assign_mood
from train_model import META_PATH, MODEL_PATH


class ModelNotTrainedError(FileNotFoundError):
    """Raised when the model artifacts are missing."""


def validate_features(features: dict) -> np.ndarray:
    """Check a {feature: value} mapping and return it as a 1 x n array in model order."""
    missing = [f for f in FEATURES if f not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    values = []
    for name in FEATURES:
        try:
            value = float(features[name])
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a number, got {features[name]!r}") from None
        if not np.isfinite(value):
            raise ValueError(f"'{name}' must be finite")
        if name in UNIT_FEATURES and not 0 <= value <= 1:
            raise ValueError(f"'{name}' must be between 0 and 1, got {value}")
        if name == "tempo" and value <= 0:
            raise ValueError(f"'tempo' must be positive, got {value}")
        values.append(value)
    return np.array([values])


class MoodModel:
    """Wraps the saved bundle: fitted pipeline plus the training rows' IDs and labels."""

    def __init__(self, bundle: dict, meta: dict | None = None):
        if list(bundle.get("features", [])) != FEATURES:
            raise ValueError(
                f"Model was trained on features {bundle.get('features')}, "
                f"but this code expects {FEATURES}. Retrain with: python -m train_model"
            )
        self.pipeline = bundle["pipeline"]
        self.train_track_ids = np.asarray(bundle["train_track_ids"])
        self.train_labels = np.asarray(bundle["train_labels"])
        self.meta = meta or {}
        self.k = self.pipeline.named_steps["knn"].n_neighbors

    @classmethod
    def load(cls, model_path: Path = MODEL_PATH, meta_path: Path = META_PATH) -> "MoodModel":
        # joblib files can run code when loaded: only load models you trained yourself.
        if not Path(model_path).exists():
            raise ModelNotTrainedError(
                f"Model not found: {model_path}. Train it with: python -m train_model"
            )
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8")) if Path(meta_path).exists() else {}
        return cls(joblib.load(model_path), meta)

    def predict(self, features: dict) -> dict:
        X = validate_features(features)
        probs = self.pipeline.predict_proba(X)[0]
        by_class = dict(zip(self.pipeline.classes_, probs))
        probabilities = {m: float(by_class.get(m, 0.0)) for m in MOODS}
        mood = str(self.pipeline.predict(X)[0])
        return {
            "mood": mood,
            "probabilities": probabilities,
            "rule_mood": rule_mood(features),
        }

    def explain(self, features: dict, exclude_track_id: str | None = None) -> dict:
        """Return the k nearest training songs and how they voted.

        If the song being explained is itself in the training set, it is its own
        nearest neighbour; pass its id as exclude_track_id to see the other k.
        """
        X = validate_features(features)
        scaled = self.pipeline.named_steps["scaler"].transform(X)
        knn = self.pipeline.named_steps["knn"]
        n = self.k + 1 if exclude_track_id is not None else self.k
        distances, indices = knn.kneighbors(scaled, n_neighbors=min(n, len(self.train_track_ids)))
        rows = [
            {"track_id": str(self.train_track_ids[i]), "mood": str(self.train_labels[i]), "distance": float(d)}
            for d, i in zip(distances[0], indices[0])
            if str(self.train_track_ids[i]) != exclude_track_id
        ][: self.k]
        votes = Counter(r["mood"] for r in rows)
        return {"neighbours": rows, "votes": {m: votes.get(m, 0) for m in MOODS}}


def rule_mood(features: dict) -> str:
    """The label the labelling rules would give, for comparison with the model."""
    return str(assign_mood(features["valence"], features["energy"]))


def distance_to_boundary(features: dict) -> float:
    """How far the song sits from the nearest 0.5 rule boundary (0 = on it)."""
    return float(min(abs(features["valence"] - THRESHOLD), abs(features["energy"] - THRESHOLD)))


def neighbours_frame(explanation: dict, catalog: pd.DataFrame) -> pd.DataFrame:
    """Join neighbour IDs to catalog rows for display."""
    nb = pd.DataFrame(explanation["neighbours"])
    if nb.empty:
        return nb
    info = catalog.set_index("track_id")[["track_name", "artists", *FEATURES]]
    return nb.join(info, on="track_id")
