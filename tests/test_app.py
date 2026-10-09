"""Smoke-test the real Streamlit app headlessly (needs data/songs.csv and the trained model).

User songs are redirected to a temporary file, so these tests never touch data/user_songs.csv.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from data_layer import SONGS_PATH, load_catalog
from train_model import MODEL_PATH
from user_songs import load_user_songs

APP = str(Path(__file__).resolve().parents[1] / "app.py")
pytestmark = pytest.mark.skipif(
    not (SONGS_PATH.exists() and MODEL_PATH.exists()),
    reason="build the catalog and train the model first",
)


@pytest.fixture(autouse=True)
def isolated_user_songs(tmp_path, monkeypatch):
    path = tmp_path / "user_songs.csv"
    monkeypatch.setenv("MOODLENS_USER_SONGS", str(path))
    return path


@pytest.fixture(scope="module")
def some_track_id():
    return load_catalog()["track_id"].iloc[0]


def pick(key: str, track_id: str) -> dict:
    """The state streamlit-searchbox keeps (under `key`) after the user picks a suggestion."""
    return {key: {"result": track_id, "search": "x", "options_js": [], "key_react": f"{key}_react_test"}}


def run_app(**session) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    for key, value in session.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


def has_text(at: AppTest, text: str) -> bool:
    return any(text in m.value for m in at.markdown)


def test_app_starts():
    at = run_app()
    assert has_text(at, "MoodLens")


def test_slider_prediction():
    at = run_app(predict_mode="🎚️ Set the sliders")
    assert has_text(at, "Predicted mood")


def test_searched_song_prediction_and_journey(some_track_id):
    at = run_app(**pick("predict_search", some_track_id), **pick("journey_search", some_track_id))
    assert has_text(at, "Predicted mood") and has_text(at, "Why this mood?")


def test_mood_control_preset():
    at = run_app()
    at.button(key="preset_🌧️ Let it hurt").click().run()
    assert not at.exception, at.exception
    assert at.session_state["ctl_v"] == (0.0, 0.35)


def test_manual_song_is_saved(isolated_user_songs):
    at = run_app(add_mode="✍️ Enter features")
    at.text_input(key="man_title").set_value("Test Tune")
    at.text_input(key="man_artist").set_value("Test Artist")
    at.button(key="man_save").click().run()
    assert not at.exception, at.exception
    saved = load_user_songs(isolated_user_songs)
    assert list(saved["track_name"]) == ["Test Tune"]
    assert has_text(at, "★ My library (1)")


def test_public_mode_keeps_songs_in_the_session(isolated_user_songs, monkeypatch):
    monkeypatch.setenv("MOODLENS_LIBRARY", "session")
    at = run_app(add_mode="✍️ Enter features")
    at.text_input(key="man_title").set_value("Visitor Tune")
    at.text_input(key="man_artist").set_value("Visitor")
    at.button(key="man_save").click().run()
    assert not at.exception, at.exception
    assert has_text(at, "★ My library (1)")
    assert not isolated_user_songs.exists()          # nothing written to disk
    fresh_visitor = run_app(add_mode="✍️ Enter features")
    assert has_text(fresh_visitor, "★ My library (0)")   # another session sees none of it


def test_manual_song_requires_title(isolated_user_songs):
    at = run_app(add_mode="✍️ Enter features")
    at.button(key="man_save").click().run()
    assert any("Title is required" in e.value for e in at.error)
    assert not isolated_user_songs.exists()
