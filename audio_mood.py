"""Predict valence and energy for an uploaded audio file with the DEAM-trained model.

Trained and benchmarked by train_audio_mood.py on 1,802 songs rated by listeners.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import joblib
import numpy as np

from audio_embeddings import (SR_CLAP, SR_FEATURES, ClapEmbedder, librosa_feature_names, librosa_features,
                              load_audio, middle_segment)
from train_audio_mood import LITE_MODEL_PATH, META_PATH, MODEL_PATH

MAX_SECONDS = 180


def clap_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


class AudioModelNotTrainedError(FileNotFoundError):
    pass


class AudioMoodModel:
    def __init__(self, bundle: dict, meta: dict | None = None):
        if bundle.get("librosa_features") != librosa_feature_names():
            raise ValueError("Audio model feature schema changed. Retrain with: python -m train_audio_mood")
        self.feature_set = bundle["feature_set"]
        self.models = bundle["models"]
        self.meta = meta or {}
        self._clap = None

    @classmethod
    def load(cls, path: Path | None = None, meta_path: Path = META_PATH) -> "AudioMoodModel":
        """Load the best model, or the CLAP-free lite model when PyTorch/transformers are missing."""
        if path is None:
            path = MODEL_PATH if clap_available() and MODEL_PATH.exists() else LITE_MODEL_PATH
        if not Path(path).exists():
            raise AudioModelNotTrainedError(
                f"Audio mood model not found: {path}. Train it with: python -m train_audio_mood")
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8")) if Path(meta_path).exists() else {}
        model = cls(joblib.load(path), meta)
        if model.uses_clap and not clap_available():
            raise AudioModelNotTrainedError("This audio model needs torch and transformers (requirements-dev.txt).")
        return model

    @property
    def uses_clap(self) -> bool:
        return self.feature_set in {"clap", "both"}

    def _features(self, data: bytes) -> np.ndarray:
        parts = []
        if self.feature_set in {"librosa", "both"}:
            y = middle_segment(load_audio(io.BytesIO(data), SR_FEATURES, MAX_SECONDS), SR_FEATURES)
            parts.append(librosa_features(y, SR_FEATURES))
        if self.uses_clap:
            if self._clap is None:
                self._clap = ClapEmbedder()
            y48 = middle_segment(load_audio(io.BytesIO(data), SR_CLAP, MAX_SECONDS), SR_CLAP)
            parts.append(self._clap.embed(y48))
        return np.concatenate(parts)[None, :]

    def predict(self, data: bytes) -> dict:
        X = self._features(data)
        return {t: float(np.clip(m.predict(X)[0], 0, 1)) for t, m in self.models.items()}

    def quality(self) -> dict:
        """Cross-validated scores of the chosen method, plus the old pipeline for comparison."""
        results = self.meta.get("results", {})
        return {"chosen": results.get(self.feature_set, {}), "old": results.get("old", {}),
                "n_songs": self.meta.get("n_songs")}
