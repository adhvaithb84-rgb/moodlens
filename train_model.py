"""Train and evaluate the MoodLens KNN classifier, then save it.

    python -m train_model            # k = 5
    python -m train_model --k 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_layer import FEATURES, MOODS, SONGS_PATH, THRESHOLD, load_catalog

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "mood_knn.joblib"
META_PATH = MODELS_DIR / "model_meta.json"

DEFAULT_K = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
BOUNDARY_MARGIN = 0.05  # "near the rule boundary" = within this of 0.5 on valence or energy


def build_pipeline(k: int = DEFAULT_K) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])


def train_and_evaluate(catalog: pd.DataFrame, k: int = DEFAULT_K,
                       test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Split, fit on the training part only, and score on the held-out part.

    Returns (bundle, metrics). The bundle holds the fitted pipeline plus the
    training track IDs and labels, which the app needs to explain neighbours.
    """
    X = catalog[FEATURES].to_numpy(dtype=float)
    y = catalog["mood"].to_numpy()
    ids = catalog["track_id"].to_numpy()

    X_train, X_test, y_train, y_test, ids_train, _ = train_test_split(
        X, y, ids, test_size=test_size, random_state=random_state, stratify=y,
    )
    if not 1 <= k <= len(X_train):
        raise ValueError(f"k={k} is invalid for {len(X_train)} training rows")

    pipeline = build_pipeline(k)
    pipeline.fit(X_train, y_train)  # the scaler only ever sees training rows
    y_pred = pipeline.predict(X_test)

    labels = [m for m in MOODS if m in set(y)]
    majority = pd.Series(y_train).mode()[0]
    near = (np.abs(X_test[:, 0] - THRESHOLD) < BOUNDARY_MARGIN) | (
        np.abs(X_test[:, 1] - THRESHOLD) < BOUNDARY_MARGIN)
    correct = y_pred == y_test

    metrics = {
        "k": k,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "train_accuracy": float(pipeline.score(X_train, y_train)),
        "majority_baseline": {"label": str(majority), "accuracy": float(np.mean(y_test == majority))},
        "report": classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": labels,
                             "matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist()},
        "boundary_analysis": {
            "margin": BOUNDARY_MARGIN,
            "near_boundary": {"n": int(near.sum()), "accuracy": float(correct[near].mean()) if near.any() else None},
            "far_from_boundary": {"n": int((~near).sum()), "accuracy": float(correct[~near].mean()) if (~near).any() else None},
        },
    }
    bundle = {"pipeline": pipeline, "train_track_ids": ids_train, "train_labels": y_train,
              "features": list(FEATURES)}
    return bundle, metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_model(bundle: dict, metrics: dict, catalog_path: Path = SONGS_PATH,
               model_path: Path = MODEL_PATH, meta_path: Path = META_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    meta = {
        "features": bundle["features"],
        "classes": [str(c) for c in bundle["pipeline"].classes_],
        "sklearn_version": sklearn.__version__,
        "catalog_sha256": _sha256(catalog_path),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "metrics": metrics,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def print_summary(metrics: dict) -> None:
    print(f"k = {metrics['k']}   train rows = {metrics['n_train']}   test rows = {metrics['n_test']}")
    print(f"Train accuracy     : {metrics['train_accuracy']:.4f}")
    print(f"Test accuracy      : {metrics['test_accuracy']:.4f}")
    base = metrics["majority_baseline"]
    print(f"Majority baseline  : {base['accuracy']:.4f}  (always '{base['label']}')")
    b = metrics["boundary_analysis"]
    print(f"Near 0.5 boundary  : {b['near_boundary']['accuracy']:.4f}  (n={b['near_boundary']['n']})")
    print(f"Far from boundary  : {b['far_from_boundary']['accuracy']:.4f}  (n={b['far_from_boundary']['n']})")
    print("\nPer class (held-out):")
    for label in metrics["confusion_matrix"]["labels"]:
        r = metrics["report"][label]
        print(f"  {label:<16} precision {r['precision']:.3f}  recall {r['recall']:.3f}  "
              f"f1 {r['f1-score']:.3f}  support {int(r['support'])}")
    print("\nConfusion matrix (rows = true, columns = predicted):")
    labels = metrics["confusion_matrix"]["labels"]
    print(" " * 17 + "".join(f"{l[:12]:>14}" for l in labels))
    for label, row in zip(labels, metrics["confusion_matrix"]["matrix"]):
        print(f"  {label:<15}" + "".join(f"{v:>14}" for v in row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="number of neighbours")
    args = parser.parse_args()

    catalog = load_catalog()
    bundle, metrics = train_and_evaluate(catalog, k=args.k)
    save_model(bundle, metrics)
    print_summary(metrics)
    print(f"\nSaved model to {MODEL_PATH}\nSaved metadata to {META_PATH}")


if __name__ == "__main__":
    main()
