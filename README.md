# MoodLens

> Don't search for a song. Search for how you want to feel.

A Streamlit app that classifies songs into four moods with a K-Nearest Neighbors
model and lets you explore 81,061 songs by feeling.

## Quick start (Windows, PowerShell)

```powershell
cd "E:\java\spotify project\moodlens"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # or requirements.txt for the lite setup

# Only needed on a fresh checkout (data/raw is not committed — see data/README.md)
.\.venv\Scripts\python.exe -m data_layer      # raw CSV -> data/songs.csv
.\.venv\Scripts\python.exe -m train_model     # -> models/mood_knn.joblib + model_meta.json
.\.venv\Scripts\python.exe -m train_estimator # -> models/feature_estimator.joblib (danceability for uploads)
.\.venv\Scripts\python.exe -m train_audio_mood # -> models/audio_mood.joblib (needs DEAM in data/deam, ~30 min)

.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pytest
```

## Features

| Tab | Feature | How it works | Uses the ML model? |
|---|---|---|---|
| Predict | Mood Prediction | KNN (k = 5) on valence, energy, danceability, tempo | Yes |
| Predict | Explainable KNN | The 5 nearest training songs and their votes | Yes |
| Predict | Mood DNA | Radar chart of the four features | No |
| Predict | Mood Twin | Closest songs in z-scored feature space | No (distance) |
| Predict | Mood Opposite | Valence and energy mirrored through 0.5, then closest songs | No (distance) |
| Mood Universe | Valence-energy map | 4,000-song sample coloured by mood | No |
| Mood Control | Region explorer | Songs inside a valence/energy box | No (filter) |
| Mood Journey | Playlist toward a mood | Steps along a straight path; never moves away from the target | No (distance) |
| Add a song | Audio upload | Measures tempo, loudness, key; predicts valence/energy (see below) | Yes |
| Add a song | Enter features / CSV | Type the four values or upload a CSV (template provided) | Yes |

Search autocompletes as you type (2+ letters): exact title first, then titles and
artists starting with what you typed, then other matches, by popularity.

## Adding your own songs

Saved songs go to `data/user_songs.csv` (not committed), are searchable (marked ★)
and appear on the maps, but are **never used for training**.

**Audio uploads (MP3, WAV, FLAC, OGG).** Tempo, loudness and key are measured
directly. **Valence and energy** come from `train_audio_mood.py`: ridge regression
on 74 audio descriptors (timbre, brightness, rhythm, harmony) plus CLAP music
embeddings (`laion/larger_clap_music`), trained on 1,802 DEAM songs rated by
listeners. The middle 45 s of the song is analysed, matching DEAM's clip length.
Danceability is still a rough estimate from loudness/tempo/key (`train_estimator.py`).

5-fold cross-validation on DEAM (songs the model never trained on):

| Method | Valence R² | Energy R² | Positive/negative side right | Calm/intense side right | Mood right |
|---|---|---|---|---|---|
| Previous pipeline (loudness/tempo/key) | −1.70 | −2.18 | 51.6% | 52.1% | 38.3% |
| Audio descriptors | 0.45 | 0.46 | 75.6% | 78.7% | 62.1% |
| CLAP embeddings | 0.30 | 0.29 | 71.5% | 72.0% | 56.3% |
| **Descriptors + CLAP (used)** | **0.46** | **0.48** | **76.2%** | **79.1%** | **63.3%** |
| Always the most common mood | | | | | 38.1% |

Descriptors + CLAP beat descriptors alone in 10 of 10 repeated CV runs (+0.96
points on average). On a second, independent set (90 MTG-Jamendo tracks with
listener tags) the new model separates energetic from calm/sad tracks with AUC 0.86
(previously 0.78) and happy from sad with AUC 0.77 (previously 0.65).

Limits: about 1 in 3 uploads still gets the wrong mood; predictions sit close to
0.5, so songs near a boundary flip easily (the app flags these); DEAM's arousal is
not exactly Spotify's energy. The first upload after starting the app takes ~30 s
extra while the CLAP model loads.

## Model results (held-out test set, 16,213 songs)

| Metric | Value |
|---|---|
| Test accuracy | 97.24% |
| Train accuracy | 98.60% |
| Majority-class baseline | 36.35% |
| Accuracy within 0.05 of a 0.5 threshold | 86.70% (3,346 songs) |
| Accuracy elsewhere | 99.98% (12,867 songs) |

**Read this before quoting the accuracy.** The labels are computed from valence and
energy, and the model receives valence and energy as inputs. The high score means
KNN reproduces the labelling rules; nearly all errors sit next to a threshold. It
does not show that listeners feel these moods. Full metrics: `models/model_meta.json`.

## Deploying to Streamlit Community Cloud

1. Push this folder (the `moodlens` folder is the repository root) to a GitHub repository.
   `.gitignore` already leaves out `.venv`, the raw Spotify CSV, DEAM audio and your library.
2. Go to https://share.streamlit.io → **Create app** → pick the repository, branch `main`,
   main file `app.py`. Under **Advanced settings** choose Python **3.12** or **3.13**.
3. Deploy. Community Cloud installs `requirements.txt` only (no PyTorch), so audio uploads
   use the lighter descriptors-only model (62.1% vs 63.3% mood accuracy on DEAM).

On any address other than `localhost`, songs that visitors add are kept only in their own
browser session (private, lost when the tab closes). Locally they persist in
`data/user_songs.csv`. Force either mode with `MOODLENS_LIBRARY=file` or `session`.

The catalog comes from Spotify's Web API (see data/README.md); check Spotify's developer
terms before making the app public.

## Structure

```
moodlens/
├── app.py              Streamlit UI (cached catalog + model; no retraining on reruns)
├── data_layer.py       load, clean, label and validate the catalog
├── train_model.py      split, train, evaluate and save the KNN pipeline
├── model_service.py    load the model, validate input, predict, explain neighbours
├── catalog_service.py  ranked search/autocomplete, twin, opposite, region, journey
├── user_songs.py       add / validate / delete your own songs, CSV import
├── audio_features.py   measure tempo, loudness, key; danceability estimate
├── audio_embeddings.py audio descriptors + CLAP embeddings
├── audio_mood.py       DEAM-trained valence/energy model for uploads
├── train_audio_mood.py benchmark methods on DEAM, save the best
├── train_estimator.py  train + score the danceability estimator
├── data/               raw download, songs.csv, cleaning report — see data/README.md
├── models/             mood_knn.joblib (scaler + KNN), feature_estimator.joblib, metadata
├── tests/              82 tests; they use generated data and never touch data/ or models/
└── .streamlit/         dark theme, 50 MB upload limit
```

The model is saved as one scikit-learn `Pipeline` (scaler + KNN), so the scaler
and model can never get out of sync. Only load `.joblib` files you trained yourself.
