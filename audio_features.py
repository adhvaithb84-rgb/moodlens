"""Measure an uploaded audio file and estimate MoodLens's four features from it.

Measured directly: tempo, loudness, major/minor mode.
Estimated (see train_estimator.py): energy, danceability, valence.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np

from train_estimator import ESTIMATOR_META_PATH, ESTIMATOR_PATH, INPUTS, TARGETS

SUPPORTED_TYPES = ["mp3", "wav", "flac", "ogg"]
SAMPLE_RATE = 22050
MAX_SECONDS = 180      # analyse at most the first 3 minutes
MIN_SECONDS = 5

# Krumhansl-Schmuckler key profiles (C first), used to tell major from minor.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class EstimatorNotTrainedError(FileNotFoundError):
    pass


def _load(source) -> np.ndarray:
    import librosa  # imported lazily: it is slow to import and only needed for uploads

    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        y, _ = librosa.load(source, sr=SAMPLE_RATE, mono=True, duration=MAX_SECONDS)
    except Exception as exc:
        raise ValueError(f"Could not read this audio file ({type(exc).__name__}). "
                         f"Supported formats: {', '.join(SUPPORTED_TYPES)}.") from None
    if len(y) < MIN_SECONDS * SAMPLE_RATE:
        raise ValueError(f"The audio is shorter than {MIN_SECONDS} seconds.")
    if not np.any(np.abs(y) > 1e-4):
        raise ValueError("The audio is silent.")
    return y


def estimate_key(chroma_mean: np.ndarray) -> dict:
    """Best-matching key and mode from an average chroma vector."""
    scores = []
    for shift in range(12):
        for mode, profile in ((1, MAJOR_PROFILE), (0, MINOR_PROFILE)):
            scores.append((np.corrcoef(chroma_mean, np.roll(profile, shift))[0, 1], shift, mode))
    scores.sort(reverse=True)
    best = scores[0]
    best_other_mode = next(s for s in scores if s[2] != best[2])
    return {
        "key": PITCHES[best[1]],
        "mode": best[2],
        "mode_strength": float(best[0] - best_other_mode[0]),  # > 0; small = ambiguous
    }


def analyze_audio(source) -> dict:
    """Measure tempo (BPM), loudness (dBFS, like Spotify's 'loudness') and mode."""
    import librosa

    y = _load(source)
    tempo = float(np.atleast_1d(librosa.feature.tempo(y=y, sr=SAMPLE_RATE))[0])
    loudness = float(10 * np.log10(max(np.mean(y ** 2), 1e-12)))
    chroma = librosa.feature.chroma_cqt(y=y, sr=SAMPLE_RATE).mean(axis=1)
    return {
        "tempo": tempo,
        "loudness": max(loudness, -60.0),
        **estimate_key(chroma),
        "seconds_analyzed": round(len(y) / SAMPLE_RATE, 1),
    }


def quality_rating(r2: float) -> str:
    if r2 >= 0.5:
        return "fair"
    if r2 >= 0.2:
        return "weak"
    return "very weak"


class FeatureEstimator:
    def __init__(self, bundle: dict, meta: dict | None = None):
        if list(bundle["inputs"]) != INPUTS or list(bundle["targets"]) != TARGETS:
            raise ValueError("Estimator schema does not match. Retrain with: python -m train_estimator")
        self.models = bundle["models"]
        self.meta = meta or {}

    @classmethod
    def load(cls, path: Path = ESTIMATOR_PATH, meta_path: Path = ESTIMATOR_META_PATH) -> "FeatureEstimator":
        if not Path(path).exists():
            raise EstimatorNotTrainedError(
                f"Feature estimator not found: {path}. Train it with: python -m train_estimator")
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8")) if Path(meta_path).exists() else {}
        return cls(joblib.load(path), meta)

    def estimate(self, measured: dict) -> dict:
        """Four model features: tempo as measured, the rest estimated and clipped to [0, 1]."""
        import pandas as pd

        X = pd.DataFrame([[measured[i] for i in INPUTS]], columns=INPUTS)
        out = {t: float(np.clip(self.models[t].predict(X)[0], 0, 1)) for t in TARGETS}
        out["tempo"] = float(measured["tempo"])
        return out

    def quality(self) -> dict:
        metrics = self.meta.get("metrics", {})
        return {t: {**metrics[t], "rating": quality_rating(metrics[t]["r2"])} for t in TARGETS if t in metrics}
