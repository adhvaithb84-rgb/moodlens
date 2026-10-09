"""Audio analysis on generated test signals (no real music needed)."""
import io

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from audio_features import SAMPLE_RATE, FeatureEstimator, analyze_audio, quality_rating
from train_estimator import INPUTS, TARGETS, fit_estimators

NOTE_HZ = {"C": 261.63, "E": 329.63, "G": 392.00, "A": 220.00}


def to_wav(y: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def tone(freqs, seconds=8.0, amp=0.2):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return sum(amp * np.sin(2 * np.pi * f * t) for f in freqs)


def click_track(bpm=120, seconds=10.0):
    y = np.zeros(int(seconds * SAMPLE_RATE))
    click = tone([1000], seconds=0.03, amp=0.8) * np.hanning(int(0.03 * SAMPLE_RATE))
    for start in np.arange(0, seconds - 0.05, 60 / bpm):
        i = int(start * SAMPLE_RATE)
        y[i:i + len(click)] += click
    return y


def test_tempo_of_click_track():
    measured = analyze_audio(to_wav(click_track(120)))
    assert measured["tempo"] == pytest.approx(120, abs=4)


def test_loudness_of_sine():
    # A sine of amplitude 0.5 has RMS 0.5/sqrt(2), i.e. about -9.03 dBFS.
    measured = analyze_audio(to_wav(tone([440], amp=0.5)))
    assert measured["loudness"] == pytest.approx(-9.03, abs=0.1)


def test_major_and_minor_chords():
    c_major = analyze_audio(to_wav(tone([NOTE_HZ["C"], NOTE_HZ["E"], NOTE_HZ["G"]])))
    a_minor = analyze_audio(to_wav(tone([NOTE_HZ["A"], NOTE_HZ["C"], NOTE_HZ["E"]])))
    assert (c_major["key"], c_major["mode"]) == ("C", 1)
    assert (a_minor["key"], a_minor["mode"]) == ("A", 0)


@pytest.mark.parametrize("data, message", [
    (b"definitely not audio", "Could not read"),
    (to_wav(tone([440], seconds=1.0)), "shorter than"),
    (to_wav(np.zeros(SAMPLE_RATE * 8)), "silent"),
], ids=["not-audio", "too-short", "silent"])
def test_bad_audio_rejected(data, message):
    with pytest.raises(ValueError, match=message):
        analyze_audio(data)


def test_estimator_fit_and_estimate():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({"loudness": rng.uniform(-30, -2, n), "tempo": rng.uniform(60, 180, n),
                       "mode": rng.integers(0, 2, n)})
    df["energy"] = np.clip((df["loudness"] + 30) / 28 + rng.normal(0, 0.05, n), 0, 1)
    df["danceability"] = rng.uniform(0, 1, n)
    df["valence"] = rng.uniform(0, 1, n)
    bundle, metrics = fit_estimators(df[INPUTS + TARGETS])

    assert metrics["energy"]["r2"] > 0.8          # learnable signal is learned
    assert metrics["valence"]["r2"] < 0.1         # pure noise is reported as such
    est = FeatureEstimator(bundle, {"metrics": metrics})
    out = est.estimate({"loudness": -5.0, "tempo": 128.0, "mode": 1})
    assert out["tempo"] == 128.0
    assert all(0 <= out[t] <= 1 for t in TARGETS)
    assert out["energy"] > 0.7
    assert est.quality()["valence"]["rating"] == "very weak"


@pytest.mark.parametrize("r2, rating", [(0.65, "fair"), (0.26, "weak"), (0.13, "very weak")])
def test_quality_rating(r2, rating):
    assert quality_rating(r2) == rating
