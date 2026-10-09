import numpy as np
import pytest

from catalog_service import MOOD_TARGETS, Catalog
from data_layer import SAD


@pytest.fixture
def catalog(catalog_df):
    return Catalog(catalog_df)


def test_search_matches_every_word_case_insensitively(catalog):
    results = catalog.search("SONG 11 artist", limit=1000)
    assert "t0011" in set(results["track_id"])
    text = (results["track_name"] + " " + results["artists"]).str.casefold()
    assert all(all(w in t for w in ["song", "11", "artist"]) for t in text)
    assert list(results["popularity"]) == sorted(results["popularity"], reverse=True)


def test_search_ranks_title_prefix_first(catalog_df):
    df = catalog_df.copy()
    df.loc[0, ["track_name", "popularity"]] = ["Dream On", 1]          # title starts with query
    df.loc[1, ["track_name", "popularity"]] = ["Sweet Dreams", 99]     # contains it, more popular
    df.loc[2, ["track_name", "popularity"]] = ["dream", 0]             # exact title
    results = Catalog(df).search("dream")
    assert list(results["track_id"][:3]) == ["t0002", "t0000", "t0001"]


def test_suggest_returns_labels_and_ids(catalog):
    suggestions = catalog.suggest("song 12")
    assert suggestions and all(isinstance(label, str) and tid.startswith("t") for label, tid in suggestions)
    assert suggestions[0] == ("Song 12 — Artist 2", "t0012")
    assert catalog.suggest("s") == []   # too short to search


def test_search_empty_and_unmatched(catalog):
    assert catalog.search("   ").empty
    assert catalog.search("zzz-not-a-song").empty


def test_get_unknown_track(catalog):
    with pytest.raises(KeyError, match="not in catalog"):
        catalog.get("missing")


def test_twin_excludes_itself_and_is_sorted(catalog):
    twins = catalog.twin("t0000", n=5)
    assert "t0000" not in set(twins["track_id"]) and len(twins) == 5
    assert list(twins["distance"]) == sorted(twins["distance"])


def test_opposite_lands_in_opposite_quadrant(catalog, catalog_df):
    far = catalog_df[(catalog_df["valence"] > 0.85) & (catalog_df["energy"] > 0.85)].iloc[0]
    opposite = catalog.opposite(far["track_id"], n=1).iloc[0]
    assert opposite["mood"] == SAD


def test_region_respects_bounds(catalog):
    region = catalog.region((0.6, 0.9), (0.1, 0.4), limit=1000)
    assert region["valence"].between(0.6, 0.9).all() and region["energy"].between(0.1, 0.4).all()
    assert catalog.region_size((0.6, 0.9), (0.1, 0.4)) == len(region)


@pytest.mark.parametrize("mood", list(MOOD_TARGETS))
def test_journey_properties(catalog, catalog_df, mood):
    start = catalog_df.iloc[0]["track_id"]
    journey = catalog.journey(start, MOOD_TARGETS[mood], steps=8)
    assert journey.iloc[0]["track_id"] == start
    assert journey["track_id"].is_unique
    assert 2 <= len(journey) <= 8
    assert list(journey["step"]) == list(range(1, len(journey) + 1))
    d = journey["distance_to_target"].to_numpy()
    assert np.all(np.diff(d) <= 1e-12), "journey must never move away from the target"


def test_journey_validation(catalog):
    with pytest.raises(ValueError):
        catalog.journey("t0000", (0.5, 0.5), steps=1)
    with pytest.raises(KeyError):
        catalog.journey("missing", (0.5, 0.5))
