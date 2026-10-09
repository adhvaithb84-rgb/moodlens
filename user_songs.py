"""Songs added by the user, stored separately from the main catalog in data/user_songs.csv.

User songs are searchable and explorable, but they are never used to train the model.
"""
from __future__ import annotations

import hashlib
import io
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_layer import (CATALOG_COLUMNS, FEATURES, USER_ID_PREFIX, _normalize_text, _read_csv,
                        assign_mood)
from model_service import validate_features

ROOT = Path(__file__).resolve().parent
DEFAULT_PATH = ROOT / "data" / "user_songs.csv"
COLUMNS = CATALOG_COLUMNS + ["source", "added_at"]
SOURCES = {"audio", "manual", "csv"}
MAX_TEXT = 200
MAX_CSV_ROWS = 5000

CSV_ALIASES = {
    "track_name": ["track_name", "title", "name", "song", "track"],
    "artists": ["artists", "artist"],
    "album_name": ["album_name", "album"],
}


def user_songs_path() -> Path:
    """Overridable with MOODLENS_USER_SONGS so tests never touch the real file."""
    return Path(os.environ.get("MOODLENS_USER_SONGS", DEFAULT_PATH))


def is_user_song(track_id: str) -> bool:
    return str(track_id).startswith(USER_ID_PREFIX)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS).astype({f: float for f in FEATURES} | {"popularity": int})


def load_user_songs(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or user_songs_path())
    if not path.exists():
        return _empty_frame()
    df = _read_csv(path)
    for col in ["album_name", "track_genres"]:
        df[col] = df[col].fillna("")
    return df[COLUMNS]


def _clean_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def make_song(title: str, artist: str, features: dict, source: str, album: str = "") -> dict:
    """Validate one song and return it as a catalog row."""
    title, artist, album = (_clean_text(x) for x in (title, artist, album))
    if not title:
        raise ValueError("Title is required")
    if not artist:
        raise ValueError("Artist is required")
    if max(len(title), len(artist), len(album)) > MAX_TEXT:
        raise ValueError(f"Title, artist and album must be at most {MAX_TEXT} characters")
    if source not in SOURCES:
        raise ValueError(f"Unknown source: {source}")
    values = dict(zip(FEATURES, validate_features(features)[0].tolist()))
    return {
        "track_id": f"{USER_ID_PREFIX}{uuid.uuid4().hex[:12]}",
        "track_name": title,
        "artists": artist,
        "album_name": album,
        "track_genres": "my uploads",
        "popularity": 0,
        **values,
        "mood": str(assign_mood(values["valence"], values["energy"])),
        "source": source,
        "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _key(df: pd.DataFrame) -> pd.Series:
    return _normalize_text(df["artists"]) + "||" + _normalize_text(df["track_name"])


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df[COLUMNS].to_csv(tmp, index=False)
    os.replace(tmp, path)  # never leaves a half-written file behind


def _merge(existing: pd.DataFrame, songs: list[dict]) -> tuple[pd.DataFrame, int, int]:
    """Append songs, skipping ones whose artist + title already exist. Returns (all, added, skipped)."""
    new = pd.DataFrame(songs, columns=COLUMNS)
    if new.empty:
        return existing, 0, 0
    seen = set(_key(existing)) if not existing.empty else set()
    keys = _key(new)
    keep = ~keys.isin(seen) & ~keys.duplicated()
    combined = pd.concat([existing, new[keep]], ignore_index=True) if not existing.empty else new[keep]
    return combined.reset_index(drop=True), int(keep.sum()), int((~keep).sum())


def add_songs(songs: list[dict], path: Path | None = None) -> tuple[int, int]:
    """Append songs to the library file. Returns (added, skipped as duplicates)."""
    path = Path(path or user_songs_path())
    combined, added, skipped = _merge(load_user_songs(path), songs)
    if added:
        _write(combined, path)
    return added, skipped


def delete_songs(track_ids: list[str], path: Path | None = None) -> int:
    path = Path(path or user_songs_path())
    existing = load_user_songs(path)
    remaining = existing[~existing["track_id"].isin(track_ids)]
    removed = len(existing) - len(remaining)
    if removed:
        _write(remaining, path)
    return removed


class FileLibrary:
    """Songs saved to data/user_songs.csv — for running MoodLens on your own computer."""

    persistent = True

    def __init__(self, path: Path | None = None):
        self.path = Path(path or user_songs_path())

    def load(self) -> pd.DataFrame:
        return load_user_songs(self.path)

    def add(self, songs: list[dict]) -> tuple[int, int]:
        return add_songs(songs, self.path)

    def delete(self, track_ids: list[str]) -> int:
        return delete_songs(track_ids, self.path)

    def version(self) -> str:
        return f"file:{self.path.stat().st_mtime_ns if self.path.exists() else 0}"


class SessionLibrary:
    """Songs kept in one visitor's session only — for public deployments.

    `store` is any dict-like object; the app passes st.session_state.
    """

    persistent = False
    KEY = "moodlens_user_songs"

    def __init__(self, store):
        self.store = store

    def load(self) -> pd.DataFrame:
        df = self.store.get(self.KEY)
        return df.copy() if df is not None else _empty_frame()

    def add(self, songs: list[dict]) -> tuple[int, int]:
        combined, added, skipped = _merge(self.load(), songs)
        if added:
            self.store[self.KEY] = combined
        return added, skipped

    def delete(self, track_ids: list[str]) -> int:
        existing = self.load()
        remaining = existing[~existing["track_id"].isin(track_ids)].reset_index(drop=True)
        self.store[self.KEY] = remaining
        return len(existing) - len(remaining)

    def version(self) -> str:
        ids = "|".join(self.load()["track_id"])
        return "session:" + hashlib.sha1(ids.encode()).hexdigest()


def parse_csv(data: bytes) -> tuple[list[dict], list[str]]:
    """Read an uploaded CSV of songs. Returns (valid songs, error messages)."""
    try:
        df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, na_values=[""])
    except Exception as exc:
        return [], [f"Could not read the CSV: {exc}"]
    df.columns = [str(c).strip().casefold() for c in df.columns]
    rename = {}
    for target, aliases in CSV_ALIASES.items():
        found = next((a for a in aliases if a in df.columns), None)
        if found:
            rename[found] = target
    df = df.rename(columns=rename)
    missing = [c for c in ["track_name", "artists", *FEATURES] if c not in df.columns]
    if missing:
        return [], [f"Missing columns: {', '.join(missing)}. Download the template to see the format."]
    if len(df) > MAX_CSV_ROWS:
        return [], [f"Too many rows ({len(df):,}); the limit is {MAX_CSV_ROWS:,}."]

    songs, errors = [], []
    for i, row in df.iterrows():
        try:
            songs.append(make_song(row["track_name"], row["artists"], {f: row[f] for f in FEATURES},
                                   source="csv", album=row.get("album_name", "")))
        except ValueError as exc:
            errors.append(f"Row {i + 2}: {exc}")  # +2: header line and 1-based numbering
    return songs, errors


def csv_template() -> bytes:
    example = pd.DataFrame([
        {"track_name": "Example Song", "artists": "Example Artist", "album_name": "",
         "valence": 0.72, "energy": 0.65, "danceability": 0.58, "tempo": 118},
    ])
    return example.to_csv(index=False).encode("utf-8")
