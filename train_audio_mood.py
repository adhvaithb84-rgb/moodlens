"""Train and benchmark the audio -> valence/energy models on DEAM.

DEAM (MediaEval Database for Emotional Analysis of Music) has 1,802 Creative
Commons songs, each rated by several listeners for valence and arousal on a 1-9
scale. We rescale both to 0-1 ((x - 1) / 8) and treat arousal as MoodLens's energy.

Compares, with 5-fold cross-validation on songs the model has not seen:
  * old    - the previous upload pipeline (loudness/tempo/mode -> Spotify-trained estimator)
  * librosa / clap / both - ridge regression on each feature set

    python -m train_audio_mood            # extract features (cached), benchmark, save best
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audio_embeddings import (CLAP_MODEL, CLAP_REVISION, SR_CLAP, SR_FEATURES, ClapEmbedder,
                              librosa_feature_names, librosa_features, load_audio, middle_segment)

ROOT = Path(__file__).resolve().parent
DEAM_DIR = ROOT / "data" / "deam"
ANNOTATION_DIR = DEAM_DIR / "annotations" / "annotations" / "annotations averaged per song" / "song_level"
AUDIO_DIR = DEAM_DIR / "audio" / "MEMD_audio"
CACHE_PATH = DEAM_DIR / "features_cache.npz"
MODEL_PATH = ROOT / "models" / "audio_mood.joblib"
LITE_MODEL_PATH = ROOT / "models" / "audio_mood_lite.joblib"
META_PATH = ROOT / "models" / "audio_mood_meta.json"

TARGETS = {"valence": "valence_mean", "energy": "arousal_mean"}
ALPHAS = np.logspace(-2, 4, 25)
SEED = 42


def load_deam() -> pd.DataFrame:
    frames = []
    for path in sorted(ANNOTATION_DIR.glob("static_annotations_averaged_songs_*.csv")):
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df[["song_id", "valence_mean", "arousal_mean"]])
    if not frames:
        raise FileNotFoundError(f"DEAM annotations not found in {ANNOTATION_DIR}. See data/README.md.")
    df = pd.concat(frames, ignore_index=True)
    df["path"] = df["song_id"].map(lambda i: AUDIO_DIR / f"{int(i)}.mp3")
    df = df[df["path"].map(Path.exists)].reset_index(drop=True)
    for name, col in TARGETS.items():
        df[name] = (df[col] - 1) / 8
    return df


def _librosa_one(path: Path) -> np.ndarray:
    return librosa_features(middle_segment(load_audio(path, SR_FEATURES, max_seconds=180), SR_FEATURES))


def extract_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute (or load cached) librosa and CLAP features for every song."""
    ids = df["song_id"].to_numpy()
    if CACHE_PATH.exists():
        cache = np.load(CACHE_PATH)
        if np.array_equal(cache["ids"], ids):
            print("Using cached features")
            return {"librosa": cache["librosa"], "clap": cache["clap"]}

    from joblib import Parallel, delayed

    t = time.time()
    print(f"Extracting librosa features for {len(df)} songs…")
    lib = np.vstack(Parallel(n_jobs=-2, verbose=0)(delayed(_librosa_one)(p) for p in df["path"]))
    print(f"  done in {time.time() - t:.0f}s")

    t = time.time()
    print("Extracting CLAP embeddings…")
    embedder = ClapEmbedder()
    clap = []
    for i, path in enumerate(df["path"]):
        clap.append(embedder.embed(middle_segment(load_audio(path, SR_CLAP, max_seconds=180), SR_CLAP)))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(df)}  ({time.time() - t:.0f}s)")
    clap = np.vstack(clap)
    np.savez_compressed(CACHE_PATH, ids=ids, librosa=lib, clap=clap)
    return {"librosa": lib, "clap": clap}


def quadrant_scores(true_v, true_e, pred_v, pred_e) -> dict:
    tv, te, pv, pe = (np.asarray(x) >= 0.5 for x in (true_v, true_e, pred_v, pred_e))
    true_q, pred_q = tv * 2 + te, pv * 2 + pe
    majority = np.bincount(true_q).max() / len(true_q)
    return {
        "mood_accuracy": float(np.mean(true_q == pred_q)),
        "majority_baseline": float(majority),
        "valence_side_accuracy": float(np.mean(tv == pv)),
        "energy_side_accuracy": float(np.mean(te == pe)),
    }


def score(df: pd.DataFrame, preds: dict) -> dict:
    out = {}
    for t in TARGETS:
        out[t] = {"r2": float(r2_score(df[t], preds[t])),
                  "mae": float(mean_absolute_error(df[t], preds[t])),
                  "pearson": float(np.corrcoef(df[t], preds[t])[0, 1])}
    out.update(quadrant_scores(df["valence"], df["energy"], preds["valence"], preds["energy"]))
    return out


def make_model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def old_pipeline_predictions(lib: np.ndarray) -> dict:
    """What the previous upload pipeline would have predicted for each DEAM song."""
    from audio_features import FeatureEstimator

    names = librosa_feature_names()
    col = {n: i for i, n in enumerate(names)}
    est = FeatureEstimator.load()
    rows = [est.estimate({"loudness": r[col["loudness_db"]], "tempo": r[col["tempo"]],
                          "mode": int(r[col["mode_major"]])}) for r in lib]
    return {t: np.array([r[t] for r in rows]) for t in TARGETS}


def main() -> None:
    df = load_deam()
    print(f"DEAM songs with audio: {len(df)}")
    feats = extract_features(df)
    feats["both"] = np.hstack([feats["librosa"], feats["clap"]])

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {"old": score(df, old_pipeline_predictions(feats["librosa"]))}
    for name in ["librosa", "clap", "both"]:
        preds = {t: np.clip(cross_val_predict(make_model(), feats[name], df[t], cv=cv), 0, 1) for t in TARGETS}
        results[name] = score(df, preds)

    print(f"\n5-fold cross-validated results on {len(df)} DEAM songs (old = no training on DEAM):")
    print(f"{'method':<9}{'valence R2':>11}{'energy R2':>11}{'valence side':>14}{'energy side':>13}{'mood (4-way)':>14}")
    for name, r in results.items():
        print(f"{name:<9}{r['valence']['r2']:>11.3f}{r['energy']['r2']:>11.3f}{r['valence_side_accuracy']:>14.1%}"
              f"{r['energy_side_accuracy']:>13.1%}{r['mood_accuracy']:>14.1%}")
    print(f"Majority-mood baseline: {results['old']['majority_baseline']:.1%}")

    best = max(["librosa", "clap", "both"],
               key=lambda n: results[n]["valence"]["r2"] + results[n]["energy"]["r2"])
    print(f"\nBest feature set: {best}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # The best model, plus a "lite" one without CLAP for deployments that can't run PyTorch.
    for feature_set, path in [(best, MODEL_PATH), ("librosa", LITE_MODEL_PATH)]:
        models = {t: make_model().fit(feats[feature_set], df[t]) for t in TARGETS}
        joblib.dump({"feature_set": feature_set, "models": models, "targets": list(TARGETS),
                     "librosa_features": librosa_feature_names()}, path)
        print(f"Saved {path} ({feature_set})")
    META_PATH.write_text(json.dumps({
        "feature_set": best,
        "clap_model": CLAP_MODEL, "clap_revision": CLAP_REVISION,
        "dataset": "DEAM (MediaEval 2015-2017), static song-level averages, rescaled (x-1)/8",
        "n_songs": int(len(df)),
        "evaluation": "5-fold cross-validation (KFold, shuffle, seed 42)",
        "results": results,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
