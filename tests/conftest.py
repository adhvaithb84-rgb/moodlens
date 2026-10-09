"""Shared fixtures. Tests use small generated catalogs and never touch data/ or models/."""
import numpy as np
import pandas as pd
import pytest

from data_layer import CATALOG_COLUMNS, assign_mood


def make_catalog(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A test-only catalog with random (not real) feature values."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "track_id": [f"t{i:04d}" for i in range(n)],
        "track_name": [f"Song {i}" for i in range(n)],
        "artists": [f"Artist {i % 10}" for i in range(n)],
        "album_name": "Test Album",
        "track_genres": "test",
        "popularity": rng.integers(0, 100, n),
        "valence": rng.uniform(0, 1, n).round(3),
        "energy": rng.uniform(0, 1, n).round(3),
        "danceability": rng.uniform(0, 1, n).round(3),
        "tempo": rng.uniform(60, 180, n).round(1),
    })
    df["mood"] = assign_mood(df["valence"], df["energy"])
    return df[CATALOG_COLUMNS]


@pytest.fixture
def catalog_df() -> pd.DataFrame:
    return make_catalog()
