import pandas as pd
import pytest

from data_layer import (ANGRY, CALM, HAPPY, SAD, _read_csv, assign_mood, clean_catalog,
                        load_catalog, validate_catalog)


@pytest.mark.parametrize("valence, energy, expected", [
    (0.5, 0.5, HAPPY),      # the threshold itself counts as "high"
    (0.9, 0.1, CALM),
    (0.5, 0.499, CALM),
    (0.499, 0.5, ANGRY),
    (0.0, 0.0, SAD),
    (0.499, 0.499, SAD),
])
def test_assign_mood_rules(valence, energy, expected):
    assert assign_mood(valence, energy) == expected


def test_assign_mood_vectorised():
    assert list(assign_mood([0.9, 0.1], [0.9, 0.1])) == [HAPPY, SAD]


def raw_row(track_id, name="Song", artist="Artist", genre="pop", popularity=50,
            valence=0.7, energy=0.7, danceability=0.5, tempo=120.0):
    return dict(track_id=track_id, track_name=name, artists=artist, album_name="Album",
                popularity=popularity, track_genre=genre, valence=valence, energy=energy,
                danceability=danceability, tempo=tempo)


def test_clean_catalog_merges_dedupes_and_drops_invalid():
    raw = pd.DataFrame([
        raw_row("a", genre="pop"),
        raw_row("a", genre="rock"),                                   # same id, second genre
        raw_row("b", name="Other", tempo=0.0, valence=0.0),          # invalid audio features
        raw_row("c", name=None),                                      # no identity
        raw_row("d", name="Hello", artist="Adele", popularity=10),
        raw_row("e", name="hello ", artist="ADELE", popularity=90),  # same song, other release
    ])
    df, report = clean_catalog(raw)

    assert sorted(df["track_id"]) == ["a", "e"]
    assert df.set_index("track_id").loc["a", "track_genres"] == "pop;rock"
    assert report["merged_genre_duplicates"] == 1
    assert report["dropped_invalid_features"] == 1
    assert report["dropped_missing_identity"] == 1
    assert report["dropped_same_song_versions"] == 1
    validate_catalog(df)


def test_clean_catalog_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        clean_catalog(pd.DataFrame({"track_id": ["a"]}))


def test_songs_titled_na_survive_csv_round_trip(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame([raw_row("a", name="NA", artist="None")]).to_csv(path, index=False)
    df = _read_csv(path)
    assert df.loc[0, "track_name"] == "NA" and df.loc[0, "artists"] == "None"


def test_validate_catalog_accepts_good_data(catalog_df):
    validate_catalog(catalog_df)


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.assign(valence=d["valence"].where(d.index != 0, 1.5)), "outside"),
    (lambda d: d.assign(tempo=d["tempo"].where(d.index != 0, 0)), "tempo"),
    (lambda d: pd.concat([d, d.head(1)]), "duplicate track_id"),
    (lambda d: d.assign(mood=d["mood"].where(d.index != 0, "Ecstatic")), "unknown mood"),
    (lambda d: d.assign(mood=SAD, valence=0.9, energy=0.9), "disagree"),
    (lambda d: d.assign(track_name=d["track_name"].where(d.index != 0, None)), "missing values"),
])
def test_validate_catalog_reports_problems(catalog_df, mutate, message):
    with pytest.raises(ValueError, match=message):
        validate_catalog(mutate(catalog_df))


def test_load_catalog_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m data_layer"):
        load_catalog(tmp_path / "nope.csv")


def test_load_catalog_round_trip(tmp_path, catalog_df):
    path = tmp_path / "songs.csv"
    catalog_df.to_csv(path, index=False)
    loaded = load_catalog(path)
    assert len(loaded) == len(catalog_df)
    assert loaded["track_id"].dtype == object or str(loaded["track_id"].dtype).startswith("str")
