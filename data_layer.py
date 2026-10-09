"""Load, clean, label and validate the MoodLens song catalog.

Build the catalog from the raw download:
    python -m data_layer
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "spotify_tracks.csv"
SONGS_PATH = ROOT / "data" / "songs.csv"
REPORT_PATH = ROOT / "data" / "cleaning_report.json"

# Model input features, in the order the model sees them.
FEATURES = ["valence", "energy", "danceability", "tempo"]
UNIT_FEATURES = ["valence", "energy", "danceability"]  # must lie in [0, 1]

HAPPY = "Happy / Excited"
CALM = "Calm / Content"
ANGRY = "Angry / Tense"
SAD = "Sad"
MOODS = [HAPPY, CALM, ANGRY, SAD]
THRESHOLD = 0.5

USER_ID_PREFIX = "user-"  # track_ids of songs the user added (see user_songs.py)

CATALOG_COLUMNS = [
    "track_id", "track_name", "artists", "album_name", "track_genres",
    "popularity", *FEATURES, "mood",
]


def assign_mood(valence, energy):
    """Rule-based mood label from valence and energy (scalars or arrays).

    These are provisional labels derived from two features, not verified
    human emotions.
    """
    v = np.asarray(valence, dtype=float) >= THRESHOLD
    e = np.asarray(energy, dtype=float) >= THRESHOLD
    labels = np.select([v & e, v & ~e, ~v & e], [HAPPY, CALM, ANGRY], default=SAD)
    return labels.item() if labels.ndim == 0 else labels


def _read_csv(path: Path) -> pd.DataFrame:
    # Only empty cells are missing: songs titled "NA", "None" or "null" must survive.
    return pd.read_csv(path, dtype={"track_id": str, "track_name": str, "artists": str,
                                    "album_name": str}, keep_default_na=False, na_values=[""])


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}. See data/README.md for how to obtain it."
        )
    df = _read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.casefold().str.strip().str.replace(r"\s+", " ", regex=True)


def clean_catalog(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Turn the raw download into one row per distinct song.

    Returns the cleaned catalog and a report of how many rows each step removed.
    """
    required = {"track_id", "track_name", "artists", "album_name", "popularity", "track_genre", *FEATURES}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw dataset is missing columns: {sorted(missing)}")

    report = {"raw_rows": int(len(raw))}
    df = raw.copy()

    # 1. Rows without identity cannot be searched or shown.
    before = len(df)
    df = df.dropna(subset=["track_id", "track_name", "artists"])
    report["dropped_missing_identity"] = before - len(df)

    # 2. The raw file repeats a track once per genre: merge to one row per track_id.
    before = len(df)
    genres = df.groupby("track_id")["track_genre"].agg(lambda g: ";".join(sorted(set(g))))
    df = df.drop_duplicates("track_id").set_index("track_id")
    df["track_genres"] = genres
    df = df.reset_index()
    report["merged_genre_duplicates"] = before - len(df)

    # 3. Remove rows whose audio features are missing or impossible.
    before = len(df)
    feats = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid = feats.notna().all(axis=1) & (feats["tempo"] > 0)
    for col in UNIT_FEATURES:
        valid &= feats[col].between(0, 1)
    df = df[valid].copy()
    df[FEATURES] = feats[valid]
    report["dropped_invalid_features"] = before - len(df)

    # 4. The same song released on several albums/compilations gets several IDs.
    #    Keep the most popular version so copies cannot leak across train/test.
    before = len(df)
    df["_song_key"] = _normalize_text(df["artists"]) + "||" + _normalize_text(df["track_name"])
    df = (
        df.sort_values(["popularity", "track_id"], ascending=[False, True])
        .drop_duplicates("_song_key")
        .drop(columns="_song_key")
    )
    report["dropped_same_song_versions"] = before - len(df)

    df["album_name"] = df["album_name"].fillna("")
    df["mood"] = assign_mood(df["valence"], df["energy"])
    df = df[CATALOG_COLUMNS].sort_values("track_id").reset_index(drop=True)

    report["final_rows"] = int(len(df))
    report["mood_counts"] = {m: int((df["mood"] == m).sum()) for m in MOODS}
    return df, report


def validate_catalog(df: pd.DataFrame) -> None:
    """Raise ValueError listing every problem found in a catalog."""
    problems = []
    missing = set(CATALOG_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Catalog is missing columns: {sorted(missing)}")

    required = ["track_id", "track_name", "artists", *FEATURES, "mood"]
    nulls = df[required].isna().sum()
    for col, count in nulls[nulls > 0].items():
        problems.append(f"{count} missing values in '{col}'")
    dupes = int(df["track_id"].duplicated().sum())
    if dupes:
        problems.append(f"{dupes} duplicate track_id values")
    for col in UNIT_FEATURES:
        bad = int((~df[col].between(0, 1)).sum())
        if bad:
            problems.append(f"{bad} values of '{col}' outside [0, 1]")
    bad_tempo = int((~(df["tempo"] > 0)).sum())
    if bad_tempo:
        problems.append(f"{bad_tempo} non-positive tempo values")
    unknown = sorted(set(df["mood"].dropna()) - set(MOODS))
    if unknown:
        problems.append(f"unknown mood labels: {unknown}")
    elif not df[["valence", "energy"]].isna().any().any():
        mismatched = int((df["mood"] != assign_mood(df["valence"], df["energy"])).sum())
        if mismatched:
            problems.append(f"{mismatched} mood labels disagree with the valence/energy rules")

    if problems:
        raise ValueError("Invalid catalog: " + "; ".join(problems))


def load_catalog(path: Path = SONGS_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Catalog not found: {path}. Build it with: python -m data_layer")
    df = _read_csv(path)
    df["album_name"] = df["album_name"].fillna("")
    df["track_genres"] = df["track_genres"].fillna("")
    validate_catalog(df)
    return df


def build_catalog(raw_path: Path = RAW_PATH, out_path: Path = SONGS_PATH,
                  report_path: Path = REPORT_PATH) -> dict:
    df, report = clean_catalog(load_raw(raw_path))
    validate_catalog(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_catalog(), indent=2))
