from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from .audio_projection import _determine_mood, project_to_cinematic


@dataclass(frozen=True)
class MovieRec:
    title: str
    overview: str
    score: float


class MusicToMovieRecommender:
    """
    Hybrid recommender:
    - FAISS similarity over tone embeddings (movie_embeddings)
    - plus alignment between track cinematic vectors and movie cinematic vectors (PCA->6D)
    """

    def __init__(self, tmdb_df, faiss_index, movie_embeddings, sentence_model):
        self.tmdb_df = tmdb_df.reset_index(drop=True)
        self.index = faiss_index
        self.movie_embeddings = np.asarray(movie_embeddings, dtype="float32")
        self.sentence_model = sentence_model

        # Precompute movie cinematic vectors using PCA->6D, then MinMax normalize per column.
        pca = PCA(n_components=6, random_state=42)
        movie_cv = pca.fit_transform(self.movie_embeddings)
        movie_cv = np.asarray(movie_cv, dtype="float32")
        scaler = MinMaxScaler()
        movie_cv = scaler.fit_transform(movie_cv)
        self.movie_cinematic_vectors = np.asarray(movie_cv, dtype="float32")

    def _hybrid_score(self, faiss_score, track_cinematic_vec, movie_idx) -> float:
        movie_cv = self.movie_cinematic_vectors[int(movie_idx)]
        tv = np.asarray(track_cinematic_vec, dtype="float32").reshape(-1)
        mv = np.asarray(movie_cv, dtype="float32").reshape(-1)
        alignment = float(np.dot(tv, mv) / (np.linalg.norm(tv) * np.linalg.norm(mv) + 1e-8))
        return float(0.65 * float(faiss_score) + 0.35 * alignment)

    def _extract_track_cinematic(self, track_features: dict) -> tuple[np.ndarray, str]:
        """
        Returns (cinematic_vector_6d, descriptor_string).
        """
        if not isinstance(track_features, dict):
            track_features = {}

        if "cinematic_vector" in track_features and track_features["cinematic_vector"] is not None:
            cv = np.asarray(track_features["cinematic_vector"], dtype="float32").reshape(-1)
            if cv.shape[0] != 6:
                raise ValueError("track_features['cinematic_vector'] must be length 6")
            desc = str(track_features.get("descriptor", "") or "")
            return cv, desc

        audio = track_features.get("audio_features") or track_features
        desc, cv = project_to_cinematic(audio if isinstance(audio, dict) else {})
        return cv, desc

    def recommend_for_track(self, track_features: dict, top_k: int = 5) -> list[dict]:
        # 1. FAISS search candidates
        q = np.asarray(track_features.get("text_embedding") or track_features.get("embedding"), dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        import faiss

        faiss.normalize_L2(q)
        k = max(int(top_k) * 4, int(top_k))
        distances, indices = self.index.search(q, k)
        cand_scores = distances[0].tolist()
        cand_idxs = indices[0].tolist()

        # Debugging: Log query and FAISS search results
        logging.debug(f"Query vector (q): {q}")
        logging.debug(f"FAISS distances: {distances}")
        logging.debug(f"FAISS indices: {indices}")

        # 2. re-rank with hybrid score
        track_cv, _desc = self._extract_track_cinematic(track_features)
        scored = []
        for i, fs in zip(cand_idxs, cand_scores):
            hybrid_score = self._hybrid_score(fs, track_cv, i)
            logging.debug(f"Movie index: {i}, FAISS score: {fs}, Hybrid score: {hybrid_score}")
            scored.append((int(i), float(fs), hybrid_score))
        scored.sort(key=lambda x: x[2], reverse=True)

        # 3. format output
        out: list[dict] = []
        for movie_idx, faiss_score, hybrid in scored[:top_k]:
            row = self.tmdb_df.iloc[int(movie_idx)]
            out.append(
                {
                    "title": str(row.get("title", "")),
                    "overview": (str(row.get("overview", "")) or "")[:180],
                    "genres": str(row.get("genres", "")),
                    "vote_average": float(row.get("vote_average", 0.0) or 0.0),
                    "original_language": str(row.get("original_language", "")),
                    "similarity_score": float(faiss_score),
                    "mood_match_score": float(hybrid),
                }
            )
        return out

    def build_profile(self, music_features_list: list[dict]) -> tuple[np.ndarray, str]:
        cvs: list[np.ndarray] = []
        for tf in music_features_list or []:
            cv, _ = self._extract_track_cinematic(tf)
            cvs.append(cv)

        if not cvs:
            zero = np.zeros(6, dtype="float32")
            return zero, _determine_mood(zero)

        X = np.vstack(cvs).astype("float32")

        if X.shape[0] >= 3:
            km = KMeans(n_clusters=3, random_state=42, n_init="auto")
            labels = km.fit_predict(X)
            counts = Counter(labels.tolist())
            dominant = max(counts.items(), key=lambda kv: kv[1])[0]
            weights = np.ones(X.shape[0], dtype="float32")
            weights[labels == dominant] = 2.0
        else:
            weights = np.ones(X.shape[0], dtype="float32")

        profile_vec = np.average(X, axis=0, weights=weights).astype("float32")
        dominant_mood = _determine_mood(profile_vec)
        return profile_vec, dominant_mood

    def _profile_descriptor(self, music_features_list: list[dict]) -> str:
        words: list[str] = []
        for tf in music_features_list or []:
            desc = tf.get("descriptor") or tf.get("descriptor_string")
            if not desc:
                _, desc = self._extract_track_cinematic(tf)
            if desc:
                words.extend(str(desc).lower().split())
        common = [w for (w, _c) in Counter(words).most_common(12)]
        return " ".join(common).strip()

    def _genre_bucket(self, genres_value: Any) -> str:
        s = str(genres_value or "")
        # Simple bucket: first token-ish
        parts = re.split(r"[|,/]+", s)
        for p in parts:
            p = p.strip()
            if p:
                return p.lower()
        return ""

    def recommend_for_profile(self, music_features_list: list[dict], top_k: int = 10) -> list[dict]:
        profile_vec, _mood = self.build_profile(music_features_list)
        profile_desc = self._profile_descriptor(music_features_list)
        if not profile_desc:
            profile_desc = _mood

        # Encode profile descriptor -> embedding
        q = self.sentence_model.encode([profile_desc])
        q = np.asarray(q, dtype="float32")

        import faiss

        faiss.normalize_L2(q)
        k = max(int(top_k) * 6, int(top_k))
        distances, indices = self.index.search(q, k)

        # Hybrid re-rank using profile cinematic vector
        scored = []
        for i, fs in zip(indices[0].tolist(), distances[0].tolist()):
            scored.append((int(i), float(fs), self._hybrid_score(fs, profile_vec, i)))
        scored.sort(key=lambda x: x[2], reverse=True)

        # Diversity pass: max 2 per genre, no duplicate titles
        per_genre: dict[str, int] = {}
        seen_titles: set[str] = set()
        out: list[dict] = []
        for movie_idx, faiss_score, hybrid in scored:
            row = self.tmdb_df.iloc[int(movie_idx)]
            title = str(row.get("title", "")).strip()
            if not title or title.lower() in seen_titles:
                continue

            genre_bucket = self._genre_bucket(row.get("genres", ""))
            if genre_bucket:
                per_genre.setdefault(genre_bucket, 0)
                if per_genre[genre_bucket] >= 2:
                    continue

            out.append(
                {
                    "title": title,
                    "overview": (str(row.get("overview", "")) or "")[:180],
                    "genres": str(row.get("genres", "")),
                    "vote_average": float(row.get("vote_average", 0.0) or 0.0),
                    "original_language": str(row.get("original_language", "")),
                    "similarity_score": float(faiss_score),
                    "mood_match_score": float(hybrid),
                }
            )
            seen_titles.add(title.lower())
            if genre_bucket:
                per_genre[genre_bucket] += 1
            if len(out) >= top_k:
                break
        return out

    def analyze_vibe(self, music_features_list: list[dict]) -> dict:
        descriptors: list[str] = []
        track_moods: list[str] = []
        audio_feats: list[dict] = []

        for tf in music_features_list or []:
            cv, desc = self._extract_track_cinematic(tf)
            track_moods.append(_determine_mood(cv))
            if desc:
                descriptors.extend(desc.lower().split())
            if isinstance(tf, dict) and isinstance(tf.get("audio_features"), dict):
                audio_feats.append(tf["audio_features"])

        top_descriptors = [w for (w, _c) in Counter(descriptors).most_common(8)]

        audio_averages: dict[str, float] = {}
        if audio_feats:
            keys = [
                "danceability",
                "energy",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "liveness",
                "valence",
                "tempo",
            ]
            for k in keys:
                vals = []
                for d in audio_feats:
                    if k in d and d[k] is not None:
                        try:
                            vals.append(float(d[k]))
                        except Exception:
                            pass
                if vals:
                    audio_averages[k] = float(np.mean(vals))

        profile_vec, dominant_mood = self.build_profile(music_features_list)

        return {
            "dominant_mood": dominant_mood,
            "top_descriptors": top_descriptors,
            "audio_averages": audio_averages,
            "track_moods": track_moods,
        }


__all__ = ["MusicToMovieRecommender"]

