"""Turn audio into numbers a mood model can learn from.

Two feature sets are compared in train_audio_mood.py:
  * "librosa": 74 hand-crafted descriptors (timbre, brightness, rhythm, harmony, loudness)
  * "clap":    512-d embeddings from LAION's CLAP model pretrained on music
"""
from __future__ import annotations

import io

import numpy as np

SR_FEATURES = 22050
SR_CLAP = 48000
CLAP_MODEL = "laion/larger_clap_music"
CLAP_REVISION = "a0b4534a14f58e20944452dff00a22a06ce629d1"  # pinned; Apache-2.0
ANALYSIS_SECONDS = 45  # DEAM clips are 45 s; uploads use their middle 45 s to match


def middle_segment(y: np.ndarray, sr: int, seconds: float = ANALYSIS_SECONDS) -> np.ndarray:
    """The central `seconds` of a waveform (skips intros/outros of full songs)."""
    n = int(seconds * sr)
    if len(y) <= n:
        return y
    start = (len(y) - n) // 2
    return y[start:start + n]
CLAP_WINDOW_SECONDS = 10


def load_audio(source, sr: int, max_seconds: float | None = None, offset: float = 0.0) -> np.ndarray:
    """Decode bytes / file / path to a mono float waveform at `sr`."""
    import librosa

    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    y, _ = librosa.load(source, sr=sr, mono=True, offset=offset, duration=max_seconds)
    return y


def librosa_feature_names() -> list[str]:
    names = ["tempo", "beat_strength", "onset_rate", "loudness_db", "rms_mean", "rms_std",
             "zcr_mean", "zcr_std", "centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std",
             "rolloff_mean", "rolloff_std", "flatness_mean", "mode_major", "mode_strength"]
    names += [f"contrast_{i}" for i in range(7)]
    names += [f"chroma_{i}" for i in range(12)]
    names += [f"mfcc_{i}_mean" for i in range(1, 20)] + [f"mfcc_{i}_std" for i in range(1, 20)]
    return names


def librosa_features(y: np.ndarray, sr: int = SR_FEATURES) -> np.ndarray:
    import librosa

    from audio_features import estimate_key

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset_env, sr=sr))[0])
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    S = np.abs(librosa.stft(y))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr).mean(axis=1)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    key = estimate_key(chroma)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[1:]  # drop c0 (overall level; loudness covers it)

    vec = np.concatenate([
        [tempo, float(onset_env.mean()), len(onsets) / (len(y) / sr),
         10 * np.log10(max(np.mean(y ** 2), 1e-12)), rms.mean(), rms.std(),
         zcr.mean(), zcr.std(), centroid.mean(), centroid.std(), bandwidth.mean(), bandwidth.std(),
         rolloff.mean(), rolloff.std(), flatness.mean(), key["mode"], key["mode_strength"]],
        contrast, chroma, mfcc.mean(axis=1), mfcc.std(axis=1),
    ]).astype(float)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


class ClapEmbedder:
    """Averages CLAP audio embeddings over consecutive 10-second windows."""

    def __init__(self, model_name: str = CLAP_MODEL, revision: str = CLAP_REVISION):
        import torch
        from transformers import ClapModel, ClapProcessor

        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.torch = torch
        self.processor = ClapProcessor.from_pretrained(model_name, revision=revision)
        self.model = ClapModel.from_pretrained(model_name, revision=revision).eval()

    def windows(self, y: np.ndarray, max_windows: int = 6) -> list[np.ndarray]:
        n = CLAP_WINDOW_SECONDS * SR_CLAP
        chunks = [y[i:i + n] for i in range(0, max(len(y) - n // 2, 1), n)][:max_windows]
        return [c for c in chunks if len(c) >= n // 2] or [y]

    def embed(self, y_48k: np.ndarray) -> np.ndarray:
        chunks = self.windows(y_48k)
        inputs = self.processor(audio=chunks, sampling_rate=SR_CLAP, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model.get_audio_features(**inputs)
        emb = getattr(out, "pooler_output", out)
        emb = self.torch.nn.functional.normalize(emb, dim=-1).mean(dim=0)
        return emb.numpy().astype(float)
