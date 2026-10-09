"""MoodLens - search for how you want to feel.

    python -m streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_searchbox import st_searchbox

from audio_features import SUPPORTED_TYPES, EstimatorNotTrainedError, FeatureEstimator, analyze_audio
from audio_mood import AudioModelNotTrainedError, AudioMoodModel
from catalog_service import MOOD_TARGETS, Catalog, display_label
from data_layer import CATALOG_COLUMNS, FEATURES, MOODS, THRESHOLD, load_catalog, validate_catalog
from model_service import ModelNotTrainedError, MoodModel, distance_to_boundary, neighbours_frame
from user_songs import FileLibrary, SessionLibrary, csv_template, is_user_song, make_song, parse_csv

MOOD_COLORS = {
    "Happy / Excited": "#FFC94D",
    "Calm / Content": "#4DD4AC",
    "Angry / Tense": "#FF5A5F",
    "Sad": "#5B8CFF",
}
MOOD_EMOJI = {"Happy / Excited": "☀️", "Calm / Content": "🌊", "Angry / Tense": "⚡", "Sad": "🌧️"}
MOOD_BLURB = {
    "Happy / Excited": "Bright and driving — positive and full of energy.",
    "Calm / Content": "Warm but gentle — positive, low intensity.",
    "Angry / Tense": "Dark and intense — high energy, negative feel.",
    "Sad": "Low and heavy — negative feel, little energy.",
}
PRESETS = {  # Mood Control quick picks: (valence range, energy range)
    "☀️ Pump me up": ((0.60, 1.00), (0.75, 1.00)),
    "🌊 Chill out": ((0.50, 1.00), (0.00, 0.40)),
    "🌧️ Let it hurt": ((0.00, 0.35), (0.00, 0.40)),
    "⚡ Let off steam": ((0.00, 0.40), (0.75, 1.00)),
}
ACCENT = "#FF4D8D"
UNIVERSE_SAMPLE = 4000
TEMPO_RANGE = (50.0, 200.0)  # for drawing tempo on a 0-1 axis only

st.set_page_config(page_title="MoodLens", page_icon="🎧", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

[data-testid="stApp"] { background: #0B0B10; }
[data-testid="stApp"]::before {
  content: ""; position: fixed; inset: -25%; z-index: 0; pointer-events: none;
  background:
    radial-gradient(40% 35% at 15% 10%, rgba(255,77,141,.22), transparent 70%),
    radial-gradient(35% 30% at 85% 15%, rgba(139,92,246,.20), transparent 70%),
    radial-gradient(40% 35% at 70% 90%, rgba(77,212,172,.14), transparent 70%),
    radial-gradient(30% 30% at 20% 85%, rgba(91,140,255,.16), transparent 70%);
  filter: blur(30px); animation: aurora 28s ease-in-out infinite alternate;
}
@keyframes aurora { 0% {transform: translate(0,0) rotate(0deg);} 100% {transform: translate(-4%,3%) rotate(8deg);} }
[data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: transparent; position: relative; z-index: 1; }
html, body, [class*="css"], .stMarkdown, button, input { font-family: 'Space Grotesk', system-ui, sans-serif; }

/* hero */
.hero { text-align: center; padding: 1.2rem 0 .6rem; }
.logo { font-size: clamp(2.8rem, 7vw, 5rem); font-weight: 700; letter-spacing: -.04em; line-height: 1; margin: .3rem 0;
  background: linear-gradient(90deg, #FF4D8D, #8B5CF6, #5B8CFF, #4DD4AC, #FFC94D, #FF4D8D);
  background-size: 300% 100%; -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: shimmer 10s linear infinite; }
@keyframes shimmer { to { background-position: 300% 0; } }
.tagline { font-size: 1.2rem; opacity: .8; margin: .2rem 0 1rem; }
.eq { display: inline-flex; gap: 5px; align-items: flex-end; height: 34px; }
.eq span { width: 6px; border-radius: 3px; background: linear-gradient(#FF4D8D, #8B5CF6); animation: bounce 1.1s ease-in-out infinite; }
.eq span:nth-child(2){animation-delay:-.9s} .eq span:nth-child(3){animation-delay:-.6s}
.eq span:nth-child(4){animation-delay:-.3s} .eq span:nth-child(5){animation-delay:-.75s}
.eq span:nth-child(6){animation-delay:-.45s} .eq span:nth-child(7){animation-delay:-.15s}
@keyframes bounce { 0%,100% {height: 6px;} 50% {height: 34px;} }
.pills { display: flex; gap: .5rem; justify-content: center; flex-wrap: wrap; }
.pill { padding: .3rem .8rem; border-radius: 999px; font-size: .85rem; background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.1); backdrop-filter: blur(8px); }

/* tabs as pills */
.stTabs [data-baseweb="tab-list"] { gap: .35rem; background: rgba(255,255,255,.04); padding: .35rem;
  border-radius: 999px; border: 1px solid rgba(255,255,255,.08); flex-wrap: wrap; justify-content: center; }
.stTabs [data-baseweb="tab"] { border-radius: 999px; padding: .45rem 1rem; height: auto; transition: background .2s; }
.stTabs [data-baseweb="tab"]:hover { background: rgba(255,255,255,.06); }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #FF4D8D, #8B5CF6); color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* glass cards */
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,.035); border-radius: 18px !important;
  border: 1px solid rgba(255,255,255,.08) !important; backdrop-filter: blur(10px); }
[data-testid="stMetric"] { background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px; padding: .8rem 1rem; }
[data-testid="stBaseButton-primary"] { background: linear-gradient(135deg, #FF4D8D, #8B5CF6); border: 0;
  box-shadow: 0 6px 24px rgba(255,77,141,.35); transition: transform .15s, box-shadow .15s; }
[data-testid="stBaseButton-primary"]:hover { transform: translateY(-1px); box-shadow: 0 10px 30px rgba(255,77,141,.5); }
[data-testid="stBaseButton-secondary"] { border-radius: 999px; }

/* section titles */
.sec { margin: .4rem 0 .2rem; font-size: 1.35rem; font-weight: 700; }
.sec::after { content: ""; display: block; width: 48px; height: 3px; margin-top: .3rem; border-radius: 2px;
  background: linear-gradient(90deg, #FF4D8D, #8B5CF6); }
.sub { opacity: .7; font-size: .92rem; margin-bottom: .6rem; }

/* mood result card */
.mood-card { display: flex; align-items: center; gap: 1.1rem; padding: 1.2rem 1.4rem; border-radius: 22px;
  background: linear-gradient(135deg, var(--glow), rgba(255,255,255,.03));
  border: 1px solid var(--mood); box-shadow: 0 0 40px var(--glow), inset 0 0 30px rgba(255,255,255,.02);
  animation: pop .5s cubic-bezier(.2,1.4,.4,1); }
@keyframes pop { from { transform: scale(.94); opacity: 0; } }
.mood-emoji { font-size: 3.4rem; filter: drop-shadow(0 0 14px var(--mood)); animation: float 3s ease-in-out infinite; }
@keyframes float { 50% { transform: translateY(-6px); } }
.mood-main { flex: 1; min-width: 0; }
.eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; opacity: .65; }
.mood-name { font-size: clamp(1.7rem, 3.4vw, 2.5rem); font-weight: 700; color: var(--mood); line-height: 1.1; }
.mood-sub { opacity: .75; font-size: .92rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ring { width: 92px; height: 92px; border-radius: 50%; flex: none; display: grid; place-items: center;
  background: conic-gradient(var(--mood) calc(var(--p) * 1%), rgba(255,255,255,.08) 0); position: relative; }
.ring::before { content: ""; position: absolute; inset: 8px; border-radius: 50%; background: #13131B; }
.ring div { position: relative; text-align: center; line-height: 1.05; }
.ring b { font-size: 1.35rem; } .ring small { display: block; font-size: .62rem; opacity: .7; }

/* probability bars */
.bars { margin-top: .9rem; display: grid; gap: .45rem; }
.bar-row { display: grid; grid-template-columns: 9.5rem 1fr 3rem; align-items: center; gap: .6rem; font-size: .9rem; }
.bar-track { height: 12px; border-radius: 6px; background: rgba(255,255,255,.07); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; transform-origin: left; animation: grow .9s cubic-bezier(.2,.8,.2,1); }
@keyframes grow { from { transform: scaleX(0); } }
.bar-val { text-align: right; opacity: .85; font-variant-numeric: tabular-nums; }

.chip { display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .75rem; font-weight: 600;
  border: 1px solid currentColor; }
.note { padding: .7rem .9rem; border-radius: 12px; background: rgba(255,201,77,.08);
  border: 1px solid rgba(255,201,77,.3); font-size: .9rem; margin-top: .7rem; }

@media (prefers-reduced-motion: reduce) { *, *::before { animation: none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---- loading (cached across reruns) ----------------------------------------------
@st.cache_resource(show_spinner="Loading song catalog…")
def get_base_catalog() -> pd.DataFrame:
    return load_catalog()


@st.cache_resource(show_spinner=False, max_entries=32)
def get_catalog(version: str, _user: pd.DataFrame) -> tuple[Catalog, str | None]:
    """Main catalog plus the visitor's songs; rebuilt only when those songs change (`version`)."""
    base = get_base_catalog()
    user = _user
    if user.empty:
        return Catalog(base), None
    combined = pd.concat([base, user[CATALOG_COLUMNS]], ignore_index=True)
    try:
        validate_catalog(combined)
    except ValueError as exc:
        return Catalog(base), f"Your added songs were ignored because the file is invalid: {exc}"
    return Catalog(combined), None


@st.cache_resource(show_spinner="Loading model…")
def get_model() -> MoodModel:
    return MoodModel.load()


@st.cache_resource(show_spinner=False)
def get_estimator() -> FeatureEstimator | None:
    try:
        return FeatureEstimator.load()
    except EstimatorNotTrainedError:
        return None


@st.cache_resource(show_spinner=False)
def get_audio_model() -> AudioMoodModel | None:
    try:
        return AudioMoodModel.load()
    except AudioModelNotTrainedError:
        return None


@st.cache_data(show_spinner=False, max_entries=8)
def analyze_cached(digest: str, _data: bytes) -> dict:
    """Measured tempo/loudness/key plus the DEAM model's valence and energy."""
    measured = analyze_audio(_data)
    return {"measured": measured, "predicted": get_audio_model().predict(_data)}


def running_locally() -> bool:
    """True when the browser reached the app via localhost, i.e. on your own computer."""
    override = os.environ.get("MOODLENS_LIBRARY")
    if override in {"file", "session"}:
        return override == "file"
    try:
        host = st.context.headers.get("Host", "") or ""
    except Exception:
        host = ""
    return host.split(":")[0] in {"", "localhost", "127.0.0.1", "[::1]"}


# Locally, added songs persist in data/user_songs.csv. On a public deployment each
# visitor gets a private, session-only library, so nobody sees anyone else's songs.
library = FileLibrary() if running_locally() else SessionLibrary(st.session_state)

try:
    model = get_model()
    catalog, catalog_warning = get_catalog(library.version(), library.load())
except FileNotFoundError as exc:
    st.error(f"**MoodLens isn't ready yet.** {exc}")
    st.code("python -m train_model" if isinstance(exc, ModelNotTrainedError)
            else "python -m data_layer\npython -m train_model", language="powershell")
    st.stop()
except ValueError as exc:
    st.error(f"**The catalog or model failed validation.** {exc}")
    st.stop()


@st.cache_data(show_spinner=False)
def universe_sample() -> pd.DataFrame:
    base = get_base_catalog()
    return base.sample(min(UNIVERSE_SAMPLE, len(base)), random_state=0)


# ---- small helpers ------------------------------------------------------------------
def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    html(f'<div class="sec">{title}</div>' + (f'<div class="sub">{subtitle}</div>' if subtitle else ""))


SEARCHBOX_STYLE = {
    "searchbox": {
        "control": {"backgroundColor": "#17171F", "borderColor": "rgba(255,255,255,.14)", "borderRadius": 14,
                    "minHeight": 46, "boxShadow": "none"},
        "menuList": {"backgroundColor": "#17171F", "borderRadius": 12},
        "singleValue": {"color": "#F2F2F5"},
        "input": {"color": "#F2F2F5"},
        "placeholder": {"color": "#8A8A99"},
        "option": {"color": "#F2F2F5", "backgroundColor": "#17171F", "highlightColor": "#2C1F33"},
    },
    "clear": {"icon": "cross", "clearable": "always", "stroke": "#8A8A99"},
    "dropdown": {"fill": "#8A8A99"},
}


def song_search(key: str, placeholder: str) -> str | None:
    """Autocomplete search: suggestions appear as you type; returns the chosen track_id."""
    track_id = st_searchbox(catalog.suggest, placeholder=placeholder, key=key, debounce=200,
                            style_overrides=SEARCHBOX_STYLE)
    if track_id is not None and track_id not in catalog._index:
        return None  # e.g. a song that was deleted since it was picked
    return track_id


def base_figure(height: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Space Grotesk, sans-serif"))
    return fig


def universe_figure(highlight: dict | None = None, path: pd.DataFrame | None = None,
                    box: tuple | None = None, height: int = 560) -> go.Figure:
    sample = universe_sample()
    fig = base_figure(height)
    for mood in MOODS:
        part = sample[sample["mood"] == mood]
        fig.add_trace(go.Scattergl(
            x=part["valence"], y=part["energy"], mode="markers", name=f"{MOOD_EMOJI[mood]} {mood}",
            marker=dict(color=MOOD_COLORS[mood], size=5, opacity=0.5),
            text=part.apply(display_label, axis=1),
            hovertemplate="%{text}<br>valence %{x:.2f} · energy %{y:.2f}<extra></extra>",
        ))
    mine = catalog.songs[catalog.songs["track_id"].map(is_user_song)]
    if not mine.empty:
        fig.add_trace(go.Scatter(
            x=mine["valence"], y=mine["energy"], mode="markers", name="★ Your songs",
            marker=dict(size=13, color="white", symbol="star", line=dict(color=ACCENT, width=1.5)),
            text=mine.apply(display_label, axis=1), hovertemplate="%{text}<extra></extra>",
        ))
    for q_mood, (x, y) in {"Happy / Excited": (0.97, 0.97), "Calm / Content": (0.97, 0.03),
                           "Angry / Tense": (0.03, 0.97), "Sad": (0.03, 0.03)}.items():
        fig.add_annotation(x=x, y=y, text=f"{MOOD_EMOJI[q_mood]} {q_mood}", showarrow=False,
                           xanchor="right" if x > 0.5 else "left", yanchor="top" if y > 0.5 else "bottom",
                           font=dict(color=MOOD_COLORS[q_mood], size=13))
    fig.add_hline(y=THRESHOLD, line=dict(color="white", width=1, dash="dot"), opacity=0.35)
    fig.add_vline(x=THRESHOLD, line=dict(color="white", width=1, dash="dot"), opacity=0.35)
    if box:
        (v0, v1), (e0, e1) = box
        fig.add_shape(type="rect", x0=v0, x1=v1, y0=e0, y1=e1,
                      line=dict(color=ACCENT, width=2), fillcolor="rgba(255,77,141,0.14)")
    if path is not None and not path.empty:
        fig.add_trace(go.Scatter(
            x=path["valence"], y=path["energy"], mode="lines+markers+text", name="Journey",
            line=dict(color="white", width=3, shape="spline"),
            marker=dict(size=13, color=ACCENT, line=dict(color="white", width=1.5)),
            text=path["step"].astype(str), textposition="top center",
            hovertext=path.apply(display_label, axis=1), hoverinfo="text",
        ))
    if highlight:
        fig.add_trace(go.Scatter(
            x=[highlight["valence"]], y=[highlight["energy"]], mode="markers", name="Selected",
            marker=dict(size=24, color=ACCENT, symbol="star", line=dict(color="white", width=2)),
            hovertext=[highlight["label"]], hoverinfo="text",
        ))
    fig.update_xaxes(title="valence  (negative → positive)", range=[0, 1], gridcolor="rgba(255,255,255,.05)")
    fig.update_yaxes(title="energy  (calm → intense)", range=[0, 1], gridcolor="rgba(255,255,255,.05)")
    fig.update_layout(legend=dict(orientation="h", y=1.1))
    return fig


def mood_dna_figure(features: dict, color: str) -> go.Figure:
    tempo_norm = float(np.clip((features["tempo"] - TEMPO_RANGE[0]) / (TEMPO_RANGE[1] - TEMPO_RANGE[0]), 0, 1))
    axes = ["valence", "energy", "danceability", "tempo"]
    values = [features["valence"], features["energy"], features["danceability"], tempo_norm]
    shown = [f"{features['valence']:.2f}", f"{features['energy']:.2f}",
             f"{features['danceability']:.2f}", f"{features['tempo']:.0f} BPM"]
    fig = base_figure(330)
    fig.add_trace(go.Scatterpolar(
        r=values + values[:1], theta=axes + axes[:1], fill="toself",
        line=dict(color=color, width=3), fillcolor=color + "40",
        customdata=shown + shown[:1], hovertemplate="%{theta}: %{customdata}<extra></extra>",
    ))
    fig.update_layout(showlegend=False, polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="rgba(255,255,255,.12)"),
        angularaxis=dict(gridcolor="rgba(255,255,255,.12)")))
    return fig


def mood_card(mood: str, label: str, agree: int, k: int) -> str:
    color = MOOD_COLORS[mood]
    return f"""
    <div class="mood-card" style="--mood:{color}; --glow:{color}33;">
      <div class="mood-emoji">{MOOD_EMOJI[mood]}</div>
      <div class="mood-main">
        <div class="eyebrow">Predicted mood</div>
        <div class="mood-name">{escape(mood)}</div>
        <div class="mood-sub" title="{escape(label)}">{escape(label)}</div>
        <div class="mood-sub">{MOOD_BLURB[mood]}</div>
      </div>
      <div class="ring" style="--p:{agree / k * 100:.0f}"><div><b>{agree}/{k}</b><small>neighbours<br>agree</small></div></div>
    </div>"""


def probability_bars(probabilities: dict) -> str:
    rows = "".join(
        f'<div class="bar-row"><span>{MOOD_EMOJI[m]} {escape(m)}</span>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{probabilities[m] * 100:.1f}%;'
        f'background:linear-gradient(90deg,{MOOD_COLORS[m]}66,{MOOD_COLORS[m]})"></div></div>'
        f'<span class="bar-val">{probabilities[m]:.0%}</span></div>'
        for m in MOODS
    )
    return f'<div class="bars">{rows}</div>'


def song_table(df: pd.DataFrame, extra: list[str] | None = None) -> None:
    view = df.assign(
        song=df.apply(display_label, axis=1),
        mood=df["mood"].map(lambda m: f"{MOOD_EMOJI.get(m, '')} {m}"),
    )[["song", "mood", *FEATURES, *(extra or [])]]
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "song": st.column_config.TextColumn("Song", width="large"), "mood": "Mood",
        "valence": st.column_config.ProgressColumn("valence", min_value=0, max_value=1, format="%.2f"),
        "energy": st.column_config.ProgressColumn("energy", min_value=0, max_value=1, format="%.2f"),
        "danceability": st.column_config.ProgressColumn("dance", min_value=0, max_value=1, format="%.2f"),
        "tempo": st.column_config.NumberColumn("BPM", format="%.0f"),
        "distance": st.column_config.NumberColumn("distance", format="%.3f"),
        "distance_to_target": st.column_config.NumberColumn("to target", format="%.3f"),
    })


def render_analysis(features: dict, label: str, track_id: str | None = None, similar: bool = True) -> dict:
    """Prediction card, Mood DNA, neighbour explanation, and optionally Twin/Opposite."""
    result = model.predict(features)
    explanation = model.explain(features, exclude_track_id=track_id)
    mood = result["mood"]

    left, right = st.columns([1.15, 1])
    with left:
        html(mood_card(mood, label, explanation["votes"][mood], model.k) + probability_bars(result["probabilities"]))
        margin = distance_to_boundary(features)
        if result["rule_mood"] != mood:
            html(f'<div class="note">The labelling rules would say <b>{escape(result["rule_mood"])}</b>. '
                 f'The neighbours outvoted them because this point is only {margin:.2f} from a 0.5 boundary.</div>')
        elif margin < 0.05:
            html(f'<div class="note">Borderline: only {margin:.2f} from a 0.5 boundary — a small change in '
                 'valence or energy would flip the mood.</div>')
    with right:
        html('<div class="eyebrow" style="text-align:center">Mood DNA</div>')
        st.plotly_chart(mood_dna_figure(features, MOOD_COLORS[mood]), width="stretch")
        if track_id:
            pct = catalog.percentiles(track_id)
            st.caption(" · ".join(f"{f} > {pct[f]:.0f}% of songs" for f in FEATURES))

    with st.container(border=True):
        section("Why this mood?",
                f"KNN looks up the {model.k} most similar songs in the training set (all four features, "
                "standardised) and lets them vote.")
        vote_col, table_col = st.columns([1, 2.2])
        with vote_col:
            votes = explanation["votes"]
            fig = base_figure(250)
            fig.add_trace(go.Bar(x=[MOOD_EMOJI[m] for m in MOODS], y=[votes[m] for m in MOODS],
                                 marker=dict(color=[MOOD_COLORS[m] for m in MOODS], cornerradius=8),
                                 hovertext=MOODS, hoverinfo="text+y"))
            fig.update_yaxes(dtick=1, title="votes", range=[0, model.k])
            st.plotly_chart(fig, width="stretch")
        with table_col:
            song_table(neighbours_frame(explanation, catalog.songs), extra=["distance"])

    if similar:
        twin_col, opp_col = st.columns(2)
        with twin_col, st.container(border=True):
            section("🪞 Mood Twin", "Closest songs across all four features.")
            exclude = {track_id} if track_id else None
            song_table(catalog.nearest_to(features, n=5, exclude=exclude), extra=["distance"])
        with opp_col, st.container(border=True):
            section("🔄 Mood Opposite", "Valence and energy mirrored through 0.5; danceability and tempo kept.")
            mirrored = {**features, "valence": 1 - features["valence"], "energy": 1 - features["energy"]}
            song_table(catalog.nearest_to(mirrored, n=5, exclude=exclude), extra=["distance"])
    return result


def feature_sliders(prefix: str, defaults: dict, help_text: dict | None = None) -> dict:
    help_text = help_text or {}
    c1, c2, c3, c4 = st.columns(4)
    return {
        "valence": c1.slider("Valence", 0.0, 1.0, float(defaults["valence"]), 0.01, key=f"{prefix}_valence",
                             help=help_text.get("valence", "Musical positiveness")),
        "energy": c2.slider("Energy", 0.0, 1.0, float(defaults["energy"]), 0.01, key=f"{prefix}_energy",
                            help=help_text.get("energy", "Intensity and activity")),
        "danceability": c3.slider("Danceability", 0.0, 1.0, float(defaults["danceability"]), 0.01,
                                  key=f"{prefix}_danceability", help=help_text.get("danceability")),
        "tempo": c4.slider("Tempo (BPM)", 40.0, 250.0, float(np.clip(defaults["tempo"], 40, 250)), 1.0,
                           key=f"{prefix}_tempo", help=help_text.get("tempo")),
    }


def save_song(song: dict) -> None:
    added, _ = library.add([song])
    st.session_state["flash"] = (f"★ Saved “{song['track_name']}” to your library." if added else
                                 f"“{song['track_name']}” by {song['artists']} is already in your library.")
    st.rerun()


# ---- page --------------------------------------------------------------------------
if msg := st.session_state.pop("flash", None):
    st.toast(msg, icon="🎧")
if catalog_warning:
    st.warning(catalog_warning)

n_mine = int(catalog.songs["track_id"].map(is_user_song).sum())
html(f"""
<div class="hero">
  <div class="eq"><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
  <div class="logo">MoodLens</div>
  <div class="tagline">Don't search for a song. Search for how you want to feel.</div>
  <div class="pills">
    <span class="pill">🎵 {len(catalog) - n_mine:,} songs</span>
    <span class="pill">🧠 KNN · k = {model.k}</span>
    <span class="pill">🎨 4 moods</span>
    {f'<span class="pill">★ {n_mine} of yours</span>' if n_mine else ''}
  </div>
</div>""")

tab_predict, tab_universe, tab_control, tab_journey, tab_add, tab_model = st.tabs(
    ["🎯 Predict", "🌌 Mood Universe", "🎚️ Mood Control", "🧭 Mood Journey", "➕ Add a song", "🔬 About the model"]
)

# ---- Predict ---------------------------------------------------------------------------
with tab_predict:
    mode = st.segmented_control("Input", ["🔎 Find a song", "🎚️ Set the sliders"], default="🔎 Find a song",
                                label_visibility="collapsed", key="predict_mode")
    if mode == "🎚️ Set the sliders":
        features = feature_sliders("slider", {"valence": 0.65, "energy": 0.70, "danceability": 0.6, "tempo": 120})
        st.session_state["selected"] = {"label": "Your custom point", **features}
        render_analysis(features, "Your custom point")
    else:
        track_id = song_search("predict_search", "Start typing a song or artist… e.g. “Blinding”, “Adele”")
        if track_id:
            row = catalog.get(track_id)
            features = catalog.features(track_id)
            label = display_label(row)
            st.session_state["selected"] = {"label": label, **features}
            render_analysis(features, label, track_id=track_id)
        else:
            st.caption(f"Suggestions appear as you type (2+ letters). Searching {len(catalog):,} songs "
                       "in the local catalog — this is not a live Spotify search.")

# ---- Mood Universe -----------------------------------------------------------------------
with tab_universe:
    section("🌌 Mood Universe", f"Every dot is a song ({UNIVERSE_SAMPLE:,} sampled from {len(catalog):,}). "
            "The dotted lines are the 0.5 thresholds that define the four moods. Hover to see songs.")
    st.plotly_chart(universe_figure(highlight=st.session_state.get("selected")), width="stretch")
    if "selected" not in st.session_state:
        st.caption("Pick a song in **Predict** to place it on the map as a ★.")

# ---- Mood Control ----------------------------------------------------------------------
with tab_control:
    section("🎚️ Mood Control", "Pick a vibe or drag the ranges. Songs inside the box, most popular first.")
    st.session_state.setdefault("ctl_v", (0.6, 0.9))
    st.session_state.setdefault("ctl_e", (0.2, 0.5))

    def apply_preset(v, e):
        st.session_state["ctl_v"], st.session_state["ctl_e"] = v, e

    for col, (name, (v, e)) in zip(st.columns(len(PRESETS)), PRESETS.items()):
        col.button(name, on_click=apply_preset, args=(v, e), width="stretch", key=f"preset_{name}")
    c1, c2 = st.columns(2)
    v_range = c1.slider("Valence range", 0.0, 1.0, step=0.01, key="ctl_v")
    e_range = c2.slider("Energy range", 0.0, 1.0, step=0.01, key="ctl_e")
    map_col, list_col = st.columns([1, 1])
    with map_col:
        st.plotly_chart(universe_figure(box=(v_range, e_range), height=480), width="stretch")
    with list_col:
        total = catalog.region_size(v_range, e_range)
        st.metric("Songs in this region", f"{total:,}")
        region = catalog.region(v_range, e_range, limit=50)
        if region.empty:
            st.info("No songs here. Try widening the ranges.")
        else:
            song_table(region)

# ---- Mood Journey ---------------------------------------------------------------------
with tab_journey:
    section("🧭 Mood Journey", "Start from a song and drift step by step towards a mood. Each step is the "
            "closest unused song to the next point on the path, and never farther from the target.")
    start_id = song_search("journey_search", "Where do you start? e.g. “Someone Like You”")
    c1, c2 = st.columns([2, 1])
    target_mood = c1.segmented_control("Where do you want to end up?", MOODS, default=MOODS[0],
                                       format_func=lambda m: f"{MOOD_EMOJI[m]} {m}", key="journey_target")
    steps = c2.slider("Songs in the journey", 3, 15, 8)
    if start_id and target_mood:
        journey = catalog.journey(start_id, MOOD_TARGETS[target_mood], steps=steps)
        map_col, list_col = st.columns([1, 1])
        with map_col:
            st.plotly_chart(universe_figure(path=journey, height=480), width="stretch")
        with list_col:
            song_table(journey, extra=["distance_to_target"])
            if len(journey) < steps:
                st.info(f"Stopped after {len(journey)} songs: no unused song was closer to the target.")

# ---- Add a song ------------------------------------------------------------------------
with tab_add:
    section("➕ Add a song", "Bring your own music. Added songs are saved to your library, show up in search "
            "(marked ★) and on the maps — but are never used to train the model.")
    how = st.segmented_control("How", ["🎵 Upload audio", "✍️ Enter features", "📄 Upload CSV"],
                               default="🎵 Upload audio", label_visibility="collapsed", key="add_mode")

    if how == "🎵 Upload audio":
        estimator, audio_model = get_estimator(), get_audio_model()
        if estimator is None or audio_model is None:
            st.error("The audio models haven't been trained yet.")
            st.code("python -m train_estimator\npython -m train_audio_mood", language="powershell")
        else:
            upload = st.file_uploader("Drop an audio file", type=SUPPORTED_TYPES,
                                      help="Analyses the middle 45 seconds of the first 3 minutes. "
                                           "MP3, WAV, FLAC or OGG.")
            if upload:
                data = upload.getvalue()
                digest = hashlib.sha1(data).hexdigest()[:16]
                st.audio(data)
                analysis = None
                try:
                    spinner = ("Listening… (the first upload also loads the music model, ~30 s)"
                               if audio_model.uses_clap and audio_model._clap is None else "Listening…")
                    with st.spinner(spinner):
                        analysis = analyze_cached(digest, data)
                except ValueError as exc:
                    st.error(str(exc))
                if analysis:
                    measured, predicted = analysis["measured"], analysis["predicted"]
                    estimated = {**estimator.estimate(measured), **predicted}
                    q = audio_model.quality()["chosen"]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Tempo (measured)", f"{measured['tempo']:.0f} BPM")
                    m2.metric("Loudness (measured)", f"{measured['loudness']:.1f} dB")
                    m3.metric("Key (measured)", f"{measured['key']} {'major' if measured['mode'] else 'minor'}",
                              help=f"Major/minor certainty score {measured['mode_strength']:.2f} (small = ambiguous)")
                    dance_q = estimator.quality().get("danceability", {})
                    html(f'<div class="note"><b>How reliable is this?</b> Valence and energy come from a model '
                         f'trained on {audio_model.quality()["n_songs"]:,} songs rated by real listeners (DEAM). '
                         f'On songs it had never heard it gets the positive/negative side right '
                         f'<b>{q["valence_side_accuracy"]:.0%}</b> of the time, the calm/intense side '
                         f'<b>{q["energy_side_accuracy"]:.0%}</b>, and the exact mood <b>{q["mood_accuracy"]:.0%}</b> '
                         f'(guessing would get {q["majority_baseline"]:.0%}). Danceability is a rough estimate '
                         f'(typical error ±{dance_q.get("mae", 0):.2f}). Adjust the sliders if it feels off.</div>')
                    st.write("")
                    help_text = {
                        "valence": f"Predicted from audio (typical error ±{q['valence']['mae']:.2f})",
                        "energy": f"Predicted from audio (typical error ±{q['energy']['mae']:.2f})",
                        "danceability": f"Rough estimate (typical error ±{dance_q.get('mae', 0):.2f})",
                        "tempo": "Measured from the audio",
                    }
                    features = feature_sliders(f"aud_{digest}", estimated, help_text)
                    c1, c2 = st.columns(2)
                    title = c1.text_input("Title", value=Path(upload.name).stem, key=f"aud_{digest}_title")
                    artist = c2.text_input("Artist", value="Unknown artist", key=f"aud_{digest}_artist")
                    label = f"{title or 'Untitled'} — {artist or 'Unknown artist'}"
                    st.session_state["selected"] = {"label": label, **features}
                    render_analysis(features, label)
                    if st.button("💾 Save to my library", type="primary", key=f"aud_{digest}_save"):
                        try:
                            save_song(make_song(title, artist, features, source="audio"))
                        except ValueError as exc:
                            st.error(str(exc))

    elif how == "✍️ Enter features":
        st.caption("Know a song's Spotify-style features (e.g. from another dataset)? Enter them here.")
        c1, c2 = st.columns(2)
        title = c1.text_input("Title", key="man_title", placeholder="Song title")
        artist = c2.text_input("Artist", key="man_artist", placeholder="Artist name")
        features = feature_sliders("man", {"valence": 0.5, "energy": 0.5, "danceability": 0.5, "tempo": 120})
        label = f"{title or 'Untitled'} — {artist or 'Unknown artist'}"
        render_analysis(features, label, similar=False)
        if st.button("💾 Save to my library", type="primary", key="man_save"):
            try:
                save_song(make_song(title, artist, features, source="manual"))
            except ValueError as exc:
                st.error(str(exc))

    elif how == "📄 Upload CSV":
        st.caption("One song per row with columns: track_name (or title), artists (or artist), valence, "
                   "energy, danceability, tempo, and optionally album_name.")
        st.download_button("⬇️ Download template", csv_template(), "moodlens_template.csv", "text/csv")
        upload = st.file_uploader("Drop a CSV file", type=["csv"], key="csv_upload")
        if upload:
            songs, errors = parse_csv(upload.getvalue())
            if errors:
                with st.expander(f"⚠️ {len(errors)} row(s) skipped", expanded=not songs):
                    st.write("\n".join(f"- {e}" for e in errors[:100]))
            if songs:
                preview = pd.DataFrame(songs)
                preview["mood"] = [model.predict(s)["mood"] for s in songs]
                st.markdown(f"**{len(songs)} valid song(s)** — predicted moods:")
                song_table(preview)
                if st.button(f"💾 Add {len(songs)} song(s) to my library", type="primary", key="csv_save"):
                    added, skipped = library.add(songs)
                    st.session_state["flash"] = (f"★ Added {added} song(s)"
                                                 + (f"; skipped {skipped} already in your library." if skipped else "."))
                    st.rerun()

    mine = catalog.songs[catalog.songs["track_id"].map(is_user_song)]
    st.divider()
    section(f"★ My library ({len(mine)})",
            "" if library.persistent else "Private to you and kept only while this browser tab is open.")
    if mine.empty:
        st.caption("Nothing here yet — songs you save appear here.")
    else:
        song_table(mine)
        labels = dict(zip(mine["track_id"], mine.apply(display_label, axis=1)))
        to_delete = st.multiselect("Remove songs", list(labels), format_func=labels.get, key="delete_pick")
        if to_delete and st.button(f"🗑️ Remove {len(to_delete)} song(s)", key="delete_btn"):
            removed = library.delete(to_delete)
            st.session_state["flash"] = f"Removed {removed} song(s)."
            st.rerun()

# ---- About the model ----------------------------------------------------------------------
with tab_model:
    meta = model.meta
    metrics = meta.get("metrics", {})
    if not metrics:
        st.warning("No metrics found. Retrain with `python -m train_model`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Held-out accuracy", f"{metrics['test_accuracy']:.1%}")
        c2.metric("Majority-class baseline", f"{metrics['majority_baseline']['accuracy']:.1%}")
        b = metrics["boundary_analysis"]
        c3.metric(f"Within {b['margin']} of 0.5", f"{b['near_boundary']['accuracy']:.1%}")
        c4.metric("Elsewhere", f"{b['far_from_boundary']['accuracy']:.1%}")

        with st.container(border=True):
            section("What this number means — and what it doesn't")
            st.markdown(
                f"The mood labels are *computed* from valence and energy with fixed 0.5 thresholds, and the "
                f"model is given valence and energy as inputs. So the {metrics['test_accuracy']:.1%} accuracy "
                f"measures how well KNN reproduces those rules on {metrics['n_test']:,} held-out songs. It does "
                "**not** measure whether people actually feel these moods. Almost every mistake sits right "
                "next to a threshold."
            )
        left, right = st.columns([1.1, 1])
        with left:
            cm = metrics["confusion_matrix"]
            fig = base_figure(400)
            fig.add_trace(go.Heatmap(z=cm["matrix"], x=[MOOD_EMOJI[m] + " " + m for m in cm["labels"]],
                                     y=[MOOD_EMOJI[m] + " " + m for m in cm["labels"]],
                                     colorscale=[[0, "#13131B"], [0.5, "#8B5CF6"], [1, "#FF4D8D"]],
                                     text=cm["matrix"], texttemplate="%{text}", showscale=False))
            fig.update_xaxes(title="predicted")
            fig.update_yaxes(title="true", autorange="reversed")
            st.plotly_chart(fig, width="stretch")
        with right:
            rows = [{"Mood": f"{MOOD_EMOJI[m]} {m}", "Precision": metrics["report"][m]["precision"],
                     "Recall": metrics["report"][m]["recall"], "F1": metrics["report"][m]["f1-score"],
                     "Test songs": int(metrics["report"][m]["support"])} for m in cm["labels"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="%.3f")
                for c in ["Precision", "Recall", "F1"]})
            audio_model = get_audio_model()
            if audio_model and audio_model.meta.get("results"):
                st.markdown(f"**Audio uploads** — 5-fold cross-validation on "
                            f"{audio_model.meta['n_songs']:,} listener-rated DEAM songs:")
                names = {"old": "Previous (loudness/tempo/key)", "librosa": "Audio descriptors",
                         "clap": "CLAP music embeddings", "both": "Descriptors + CLAP"}
                st.dataframe(pd.DataFrame([
                    {"Method": names.get(n, n) + (" ✅" if n == audio_model.feature_set else ""),
                     "Valence R²": r["valence"]["r2"], "Energy R²": r["energy"]["r2"],
                     "Mood correct": r["mood_accuracy"]}
                    for n, r in audio_model.meta["results"].items()]),
                    hide_index=True, width="stretch", column_config={
                        "Valence R²": st.column_config.NumberColumn(format="%.2f"),
                        "Energy R²": st.column_config.NumberColumn(format="%.2f"),
                        "Mood correct": st.column_config.NumberColumn(format="percent")})
        st.caption(f"KNN, k = {metrics['k']}, StandardScaler, features {', '.join(meta['features'])}. "
                   f"Trained on {metrics['n_train']:,} songs ({meta.get('trained_at', '?')}). "
                   "Data: Spotify Tracks Dataset (Hugging Face, maharshipandya), deduplicated — see data/README.md.")
