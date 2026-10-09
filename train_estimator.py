"""Train the estimator that guesses Spotify-style features from measurable audio properties.

Spotify's energy, danceability and valence come from Spotify's own (unpublished)
analysis, so they cannot be computed from an audio file. What *can* be measured
from audio is loudness, tempo and major/minor mode, and the raw dataset records
those for every song. This script learns how well those three predict the other
three features and saves the result with honest held-out scores.

    python -m train_estimator
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from data_layer import RAW_PATH, load_raw

ROOT = Path(__file__).resolve().parent
ESTIMATOR_PATH = ROOT / "models" / "feature_estimator.joblib"
ESTIMATOR_META_PATH = ROOT / "models" / "estimator_meta.json"

INPUTS = ["loudness", "tempo", "mode"]
TARGETS = ["energy", "danceability", "valence"]
RANDOM_STATE = 42


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates("track_id")
    df = df[INPUTS + TARGETS].apply(pd.to_numeric, errors="coerce").dropna()
    return df[df["tempo"] > 0]


def fit_estimators(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> tuple[dict, dict]:
    train, test = train_test_split(df, test_size=0.2, random_state=random_state)
    models, metrics = {}, {}
    for target in TARGETS:
        model = HistGradientBoostingRegressor(random_state=random_state)
        model.fit(train[INPUTS], train[target])
        pred = np.clip(model.predict(test[INPUTS]), 0, 1)
        models[target] = model
        metrics[target] = {
            "r2": float(r2_score(test[target], pred)),
            "mae": float(mean_absolute_error(test[target], pred)),
            "baseline_mae": float(mean_absolute_error(test[target], np.full(len(test), train[target].mean()))),
        }
    metrics["n_train"], metrics["n_test"] = int(len(train)), int(len(test))
    return {"models": models, "inputs": INPUTS, "targets": TARGETS}, metrics


def main() -> None:
    df = prepare(load_raw(RAW_PATH))
    bundle, metrics = fit_estimators(df)
    ESTIMATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ESTIMATOR_PATH)
    ESTIMATOR_META_PATH.write_text(json.dumps({
        "inputs": INPUTS, "targets": TARGETS, "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"Held-out quality ({metrics['n_test']:,} songs):")
    for target in TARGETS:
        m = metrics[target]
        print(f"  {target:<13} R2 {m['r2']:.3f}   MAE {m['mae']:.3f}   (guessing the mean: MAE {m['baseline_mae']:.3f})")
    print(f"Saved to {ESTIMATOR_PATH}")


if __name__ == "__main__":
    main()
