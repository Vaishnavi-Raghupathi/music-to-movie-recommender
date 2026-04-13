from __future__ import annotations

import ast
import hashlib
import os
import re
from typing import Any

import faiss
import numpy as np
import pandas as pd

from .audio_projection import project_to_cinematic
from .genre_bridge import GENRE_CINEMA_MAP


def _words_from_descriptor(descriptor: str) -> set[str]:
    return {w.strip().lower() for w in (descriptor or "").split() if w.strip()}


_AUDIO_DESCRIPTOR_WORDS: set[str] = set()
for _sample in (
    {"danceability": 0.2, "energy": 0.2, "speechiness": 0.2, "acousticness": 0.8, "instrumentalness": 0.1, "liveness": 0.2, "valence": 0.2, "tempo": 80},
    {"danceability": 0.8, "energy": 0.9, "speechiness": 0.1, "acousticness": 0.1, "instrumentalness": 0.0, "liveness": 0.2, "valence": 0.9, "tempo": 170},
    {"danceability": 0.5, "energy": 0.5, "speechiness": 0.8, "acousticness": 0.2, "instrumentalness": 0.0, "liveness": 0.8, "valence": 0.3, "tempo": 130},
):
    _AUDIO_DESCRIPTOR_WORDS |= _words_from_descriptor(project_to_cinematic(_sample)[0])

_GENRE_DESCRIPTOR_WORDS: set[str] = set()
for _desc in GENRE_CINEMA_MAP.values():
    _GENRE_DESCRIPTOR_WORDS |= _words_from_descriptor(_desc)


MOOD_KEYWORDS: set[str] = {
    "melancholic",
    "atmospheric",
    "dreamlike",
    "slow-burn",
    "introspective",
    "haunting",
    "tender",
    "bittersweet",
    "euphoric",
    "desolate",
    "intimate",
    "surreal",
    "cathartic",
    "brooding",
    "nostalgic",
    "ethereal",
    "raw",
    "quiet",
    "bleak",
    "warm",
    "poignant",
    "tense",
    "contemplative",
    "lyrical",
    "visceral",
    "understated",
    "wistful",
    "alienated",
    "lonely",
    "transcendent",
    "coming-of-age",
    "existential",
    "dark",
    "hopeful",
    "yearning",
    "restless",
    "cold",
    "stark",
    "fragile",
    "turbulent",
    "meditative",
    "sensory",
    "hypnotic",
} | _AUDIO_DESCRIPTOR_WORDS | _GENRE_DESCRIPTOR_WORDS


_sentence_split_re = re.compile(r"(?<=[.!?])\s+")


def parse_json_column(value: str) -> list[str]:
    """
    Safely parse TMDB genre/keyword columns from strings like:
      \"[{'id': 28, 'name': 'Action'}]\" -> [\"Action\"]
    Returns [] on malformed input.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        s = str(value).strip()
        if not s or s.lower() in {"nan", "none"}:
            return []
        try:
            items = ast.literal_eval(s)
        except Exception:
            return []

    names: list[str] = []
    try:
        for it in items:
            if isinstance(it, dict):
                n = it.get("name")
                if n:
                    names.append(str(n))
            elif isinstance(it, str):
                names.append(it)
    except Exception:
        return []
    return names


def _tokenize_lower(text: str) -> set[str]:
    t = (text or "").lower()
    # Keep hyphenated mood words like "slow-burn", "coming-of-age"
    toks = re.findall(r"[a-z]+(?:-[a-z]+)*", t)
    return set(toks)


def build_tone_text(row: pd.Series) -> str:
    genres = parse_json_column(row.get("genres", ""))
    keywords = parse_json_column(row.get("keywords", ""))
    tagline = str(row.get("tagline", "") or "").strip()
    overview = str(row.get("overview", "") or "").strip()

    # Overview: keep only sentences containing any mood keyword
    mood_sentences: list[str] = []
    if overview:
        for sent in _sentence_split_re.split(overview):
            sent = sent.strip()
            if not sent:
                continue
            toks = _tokenize_lower(sent)
            if any(k in toks for k in MOOD_KEYWORDS):
                mood_sentences.append(sent)

    # Keywords: keep mood-relevant ones
    mood_kw = [k for k in keywords if _tokenize_lower(k) & MOOD_KEYWORDS]

    tone_text = " ".join(
        [tagline, " ".join(mood_sentences), " ".join(mood_kw), " ".join(genres)]
    ).strip()

    if len(tone_text) < 20 and overview:
        sents = [s.strip() for s in _sentence_split_re.split(overview) if s.strip()]
        tone_text = " ".join(sents[:2]).strip() or overview

    return tone_text


def apply_filters(
    tmdb_df: pd.DataFrame,
    language: str | None = None,
    genre: str | None = None,
    min_rating: float | None = None,
    min_popularity: float | None = None,
    recent_only: bool = False,
) -> pd.DataFrame:
    df = tmdb_df.copy()
    if language:
        df = df[df.get("original_language").astype(str) == str(language)]
    if genre:
        g = str(genre).lower().strip()
        if "genres" in df.columns:
            df = df[df["genres"].astype(str).str.lower().str.contains(g, na=False)]
    if min_rating is not None and str(min_rating) != "":
        if "vote_average" in df.columns:
            df = df[pd.to_numeric(df["vote_average"], errors="coerce") >= float(min_rating)]
    if min_popularity is not None and str(min_popularity) != "":
        if "popularity" in df.columns:
            df = df[pd.to_numeric(df["popularity"], errors="coerce") >= float(min_popularity)]
    if recent_only:
        if "year" in df.columns:
            df = df[pd.to_numeric(df["year"], errors="coerce") >= 2015]
        elif "release_date" in df.columns:
            years = pd.to_datetime(df["release_date"], errors="coerce").dt.year
            df = df[years >= 2015]
    return df.reset_index(drop=True)


def _df_cache_hash(df: pd.DataFrame) -> str:
    # Stable-ish hash for caching embeddings per filtered dataset.
    cols = [c for c in ["title", "tagline", "overview", "genres", "keywords", "vote_average", "popularity", "release_date", "year"] if c in df.columns]
    sub = df[cols].copy() if cols else df.copy()
    # Force deterministic string conversion for object-like cols
    for c in sub.columns:
        if sub[c].dtype == "object":
            sub[c] = sub[c].fillna("").astype(str)
    digest = hashlib.md5(pd.util.hash_pandas_object(sub, index=True).values.tobytes()).hexdigest()
    return digest


def _model_name(model: Any) -> str:
    return getattr(model, "model_card", None) or getattr(model, "name_or_path", None) or getattr(model, "model_name", None) or model.__class__.__name__


def build_movie_index(tmdb_df: pd.DataFrame, model) -> tuple[faiss.Index, np.ndarray, pd.DataFrame]:
    """
    Build/load a FAISS index for cosine similarity over tone_text embeddings.
    Returns (index, embeddings, tmdb_df_with_tone_text).
    """
    df = tmdb_df.copy()
    df["tone_text"] = df.apply(build_tone_text, axis=1)

    cache_dir = "faiss_cache"
    os.makedirs(cache_dir, exist_ok=True)
    key = _df_cache_hash(df)
    mname = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(_model_name(model)))
    index_path = os.path.join(cache_dir, f"tone_{mname}_{key}.index")
    emb_path = os.path.join(cache_dir, f"tone_{mname}_{key}.npy")

    if os.path.exists(index_path) and os.path.exists(emb_path):
        index = faiss.read_index(index_path)
        embeddings = np.load(emb_path)
        return index, embeddings, df

    tone_texts = df["tone_text"].fillna("").astype(str).tolist()
    embeddings = model.encode(tone_texts, batch_size=256, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    np.save(emb_path, embeddings)

    return index, embeddings, df


__all__ = ["build_movie_index", "apply_filters", "MOOD_KEYWORDS"]

