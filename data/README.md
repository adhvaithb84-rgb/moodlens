# MoodLens data

## Source dataset

| Field | Value |
|---|---|
| Name | Spotify Tracks Dataset |
| URL | https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset |
| Pinned revision | `635b034f69257814eff850a5c2b3346fe458134f` (file `dataset.csv`) |
| License (as stated on the dataset card) | BSD |
| Downloaded on | 2026-10-09 |
| Local file | `raw/spotify_tracks.csv` (20,118,244 bytes) |
| SHA-256 | `B202FA49909B2D5CEF71A04B1D21243CFEB36414535F2CA9272AA646721177BD` |
| Raw rows | 114,000 (89,741 unique track IDs, 125 genres) |

The dataset card says the data was collected with Spotify's Web API. The BSD
label is the uploader's claim; Spotify's own developer terms restrict how
Spotify content may be used. Treat MoodLens as an educational project and
check those terms before any public or commercial use.

To re-download the exact same file:

```powershell
Invoke-WebRequest "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/635b034f69257814eff850a5c2b3346fe458134f/dataset.csv" -OutFile data\raw\spotify_tracks.csv
```

## Cleaning (`python -m data_layer`)

Results are written to `cleaning_report.json`.

| Step | Rows removed |
|---|---|
| Missing track name / artist | 1 |
| Same track ID listed under several genres (merged; genres kept in `track_genres`) | 24,259 |
| Invalid audio features (tempo 0 with valence 0) | 157 |
| Same song (artist + title) released under several IDs — most popular kept | 8,522 |
| **Final catalog (`songs.csv`)** | **81,061 songs** |

The last step matters for evaluation: without it, copies of one song could sit
in both the training and the test set and inflate accuracy.

## DEAM (audio upload model)

| Field | Value |
|---|---|
| Name | DEAM — MediaEval Database for Emotional Analysis of Music |
| URL | https://cvml.unige.ch/databases/DEAM/ |
| Content | 1,802 songs/excerpts with listener ratings of valence and arousal (1–9) |
| License | Audio: Creative Commons (per the dataset page) |
| Downloaded on | 2026-10-09 (`DEAM_audio.zip`, `DEAM_Annotations.zip`) |
| Local folder | `deam/` (not committed) |

Used by `train_audio_mood.py`. Ratings are rescaled to 0–1 with (x − 1) / 8;
DEAM's arousal is used as MoodLens's energy. These are related but not
identical to Spotify's valence and energy.

## Mood labels

| Rule | Mood | Songs |
|---|---|---|
| valence ≥ 0.5 and energy ≥ 0.5 | Happy / Excited | 29,469 |
| valence ≥ 0.5 and energy < 0.5 | Calm / Content | 6,429 |
| valence < 0.5 and energy ≥ 0.5 | Angry / Tense | 27,740 |
| valence < 0.5 and energy < 0.5 | Sad | 17,423 |

These are provisional, rule-generated labels, **not** verified human emotions.
Calm is the smallest class (about 8%).
