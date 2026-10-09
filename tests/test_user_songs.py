import pytest

from data_layer import FEATURES, validate_catalog
from user_songs import (COLUMNS, FileLibrary, SessionLibrary, add_songs, csv_template, delete_songs,
                        is_user_song, load_user_songs, make_song, parse_csv)

FEATS = {"valence": 0.8, "energy": 0.3, "danceability": 0.5, "tempo": 100}


def test_make_song_builds_a_valid_catalog_row():
    song = make_song("  My Song ", "Me", FEATS, source="manual")
    assert is_user_song(song["track_id"])
    assert song["track_name"] == "My Song"
    assert song["mood"] == "Calm / Content"
    assert set(song) == set(COLUMNS)


@pytest.mark.parametrize("title, artist, feats, message", [
    ("", "Me", FEATS, "Title"),
    ("Song", "   ", FEATS, "Artist"),
    (float("nan"), "Me", FEATS, "Title"),
    ("x" * 201, "Me", FEATS, "at most"),
    ("Song", "Me", {**FEATS, "valence": 2}, "between 0 and 1"),
])
def test_make_song_validation(title, artist, feats, message):
    with pytest.raises(ValueError, match=message):
        make_song(title, artist, feats, source="manual")


def test_add_load_dedupe_delete(tmp_path):
    path = tmp_path / "user.csv"
    assert load_user_songs(path).empty

    a = make_song("Hello", "Adele", FEATS, source="manual")
    b = make_song("hello ", "ADELE", FEATS, source="audio")   # same song, different spelling
    c = make_song("Other", "Adele", FEATS, source="manual")
    assert add_songs([a, b], path) == (1, 1)
    assert add_songs([a, c], path) == (1, 1)

    songs = load_user_songs(path)
    assert sorted(songs["track_name"]) == ["Hello", "Other"]
    validate_catalog(songs)
    assert delete_songs([a["track_id"]], path) == 1
    assert list(load_user_songs(path)["track_name"]) == ["Other"]
    assert not list(tmp_path.glob("*.tmp"))


def test_parse_csv_template_and_aliases():
    songs, errors = parse_csv(csv_template())
    assert errors == [] and len(songs) == 1

    data = b"Title,Artist,valence,energy,danceability,tempo\nA,B,0.1,0.9,0.5,140\n"
    songs, errors = parse_csv(data)
    assert errors == [] and songs[0]["track_name"] == "A" and songs[0]["mood"] == "Angry / Tense"


def test_parse_csv_reports_bad_rows():
    data = (b"track_name,artists,valence,energy,danceability,tempo\n"
            b"Good,Me,0.5,0.5,0.5,120\n"
            b"Bad,Me,1.5,0.5,0.5,120\n"
            b",Me,0.5,0.5,0.5,120\n"
            b"Missing,Me,,0.5,0.5,120\n")
    songs, errors = parse_csv(data)
    assert [s["track_name"] for s in songs] == ["Good"]
    assert [e.split(":")[0] for e in errors] == ["Row 3", "Row 4", "Row 5"]


def test_parse_csv_missing_columns():
    songs, errors = parse_csv(b"title,artist\nA,B\n")
    assert songs == [] and "Missing columns" in errors[0]


def test_session_library_is_private_and_never_writes_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    alice, bob = SessionLibrary({}), SessionLibrary({})
    v0 = alice.version()
    a = make_song("Hello", "Adele", FEATS, source="manual")
    assert alice.add([a, make_song("hello", "adele", FEATS, source="audio")]) == (1, 1)
    assert list(alice.load()["track_name"]) == ["Hello"] and bob.load().empty
    assert alice.version() != v0
    assert alice.delete([a["track_id"]]) == 1 and alice.load().empty
    assert not any(tmp_path.iterdir())


def test_file_library_wraps_csv(tmp_path):
    lib = FileLibrary(tmp_path / "user.csv")
    v0 = lib.version()
    song = make_song("Hello", "Adele", FEATS, source="manual")
    assert lib.add([song]) == (1, 0)
    assert lib.version() != v0 and len(lib.load()) == 1
    assert lib.delete([song["track_id"]]) == 1


def test_features_survive_round_trip(tmp_path):
    path = tmp_path / "user.csv"
    song = make_song("NA", "None", FEATS, source="manual")
    add_songs([song], path)
    row = load_user_songs(path).iloc[0]
    assert row["track_name"] == "NA" and row["artists"] == "None"
    assert {f: row[f] for f in FEATURES} == pytest.approx(FEATS)
