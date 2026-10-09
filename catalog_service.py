"""Search and explore the local song catalog.

None of this needs the trained model: twins, opposites, regions and journeys are
distance calculations and filters on the catalog's own features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_layer import ANGRY, CALM, FEATURES, HAPPY, SAD, USER_ID_PREFIX

# A representative point inside each mood quadrant (valence, energy).
MOOD_TARGETS = {
    HAPPY: (0.80, 0.80),
    CALM: (0.75, 0.25),
    ANGRY: (0.20, 0.85),
    SAD: (0.20, 0.20),
}


def display_label(row: pd.Series) -> str:
    """'Title — Artist, Artist', with a star for songs the user added."""
    star = "★ " if str(row["track_id"]).startswith(USER_ID_PREFIX) else ""
    return f"{star}{row['track_name']} — {str(row['artists']).replace(';', ', ')}"


class Catalog:
    def __init__(self, songs: pd.DataFrame):
        self.songs = songs.reset_index(drop=True)
        self._index = pd.Series(self.songs.index, index=self.songs["track_id"])
        X = self.songs[FEATURES].to_numpy(dtype=float)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0
        self._Z = (X - self._mean) / self._std  # z-scores so tempo does not dominate distances
        self._ve = self.songs[["valence", "energy"]].to_numpy(dtype=float)
        self._title = self.songs["track_name"].str.casefold()
        self._artist = self.songs["artists"].str.casefold()
        self._search_text = self._title + " " + self._artist

    def __len__(self) -> int:
        return len(self.songs)

    # ---- lookup -----------------------------------------------------------
    def search(self, query: str, limit: int = 20) -> pd.DataFrame:
        """Songs whose title or artist contains every word of the query.

        Ranked for autocomplete: exact title, then title starting with the query,
        then artist starting with it, then any other match; popularity breaks ties.
        """
        q = " ".join(str(query).casefold().split())
        if not q:
            return self.songs.iloc[0:0]
        mask = np.ones(len(self.songs), dtype=bool)
        for word in q.split():
            mask &= self._search_text.str.contains(word, regex=False).to_numpy()
        hits = self.songs[mask]
        title = self._title[mask]
        rank = np.select(
            [title == q, title.str.startswith(q), self._artist[mask].str.startswith(q)],
            [0, 1, 2], default=3,
        )
        return (hits.assign(_rank=rank)
                .sort_values(["_rank", "popularity"], ascending=[True, False])
                .drop(columns="_rank").head(limit))

    def suggest(self, query: str, limit: int = 10) -> list[tuple[str, str]]:
        """(label, track_id) pairs for an autocomplete dropdown."""
        if len(str(query).strip()) < 2:
            return []
        hits = self.search(query, limit=limit)
        return list(zip(hits.apply(display_label, axis=1), hits["track_id"]))

    def get(self, track_id: str) -> pd.Series:
        if track_id not in self._index:
            raise KeyError(f"Track not in catalog: {track_id}")
        return self.songs.iloc[self._index[track_id]]

    def features(self, track_id: str) -> dict:
        row = self.get(track_id)
        return {f: float(row[f]) for f in FEATURES}

    def percentiles(self, track_id: str) -> dict:
        """Where the song ranks in the catalog for each feature (0-100)."""
        row = self.get(track_id)
        return {f: float((self.songs[f] < row[f]).mean() * 100) for f in FEATURES}

    # ---- similarity -------------------------------------------------------
    def nearest_to(self, features: dict, n: int = 5, exclude: set[str] | None = None) -> pd.DataFrame:
        """The n songs closest to a feature point, in z-scored feature space."""
        z = (np.array([features[f] for f in FEATURES], dtype=float) - self._mean) / self._std
        dist = np.linalg.norm(self._Z - z, axis=1)
        if exclude:
            dist[self.songs["track_id"].isin(exclude).to_numpy()] = np.inf
        order = np.argsort(dist, kind="stable")[:n]
        order = order[np.isfinite(dist[order])]
        return self.songs.iloc[order].assign(distance=dist[order])

    def twin(self, track_id: str, n: int = 5) -> pd.DataFrame:
        """Mood Twin: the songs most similar to this one across all four features."""
        return self.nearest_to(self.features(track_id), n=n, exclude={track_id})

    def opposite(self, track_id: str, n: int = 5) -> pd.DataFrame:
        """Mood Opposite: mirror valence and energy through 0.5, keep danceability and tempo."""
        target = self.features(track_id)
        target["valence"] = 1.0 - target["valence"]
        target["energy"] = 1.0 - target["energy"]
        return self.nearest_to(target, n=n, exclude={track_id})

    # ---- exploration ------------------------------------------------------
    def region(self, valence: tuple[float, float], energy: tuple[float, float],
               limit: int = 25) -> pd.DataFrame:
        """Mood Control: songs inside a valence/energy box, most popular first."""
        s = self.songs
        return s[self._region_mask(valence, energy)].sort_values("popularity", ascending=False).head(limit)

    def region_size(self, valence: tuple[float, float], energy: tuple[float, float]) -> int:
        return int(self._region_mask(valence, energy).sum())

    def _region_mask(self, valence, energy) -> pd.Series:
        return self.songs["valence"].between(*valence) & self.songs["energy"].between(*energy)

    def journey(self, start_id: str, target: tuple[float, float], steps: int = 8) -> pd.DataFrame:
        """Mood Journey: a playlist that moves from a start song towards a target mood.

        Waypoints are spaced evenly on the straight line from the start song to the
        target in valence/energy space. Each step picks the unused song closest to
        its waypoint, among songs no farther from the target than the previous pick,
        so the journey never moves away from the target.
        """
        if steps < 2:
            raise ValueError("A journey needs at least 2 steps")
        start = self._index.get(start_id)
        if start is None:
            raise KeyError(f"Track not in catalog: {start_id}")
        target_pt = np.array(target, dtype=float)
        to_target = np.linalg.norm(self._ve - target_pt, axis=1)
        used = np.zeros(len(self.songs), dtype=bool)
        used[start] = True
        picks = [start]
        start_pt = self._ve[start]
        for step in range(1, steps):
            waypoint = start_pt + (target_pt - start_pt) * step / (steps - 1)
            allowed = ~used & (to_target <= to_target[picks[-1]])
            if not allowed.any():
                break
            dist = np.where(allowed, np.linalg.norm(self._ve - waypoint, axis=1), np.inf)
            best = int(np.argmin(dist))
            used[best] = True
            picks.append(best)
        out = self.songs.iloc[picks].copy()
        out.insert(0, "step", range(1, len(picks) + 1))
        out["distance_to_target"] = to_target[picks]
        return out
