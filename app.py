from __future__ import annotations

import hashlib
import logging
import os

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer

import config
from core.audio_projection import project_to_cinematic
from core.genre_bridge import detect_genres_from_text, enrich_with_genre
from core.movie_index import apply_filters, build_movie_index, build_tone_text
from core.recommender import MusicToMovieRecommender
from core.spotify_client import (
    get_auth_url,
    get_spotify_client,
    get_token_from_code,
    get_top_tracks,
    refresh_if_expired,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,  # Change to logging.WARNING to reduce verbosity
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Suppress debug logs from external libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Startup (module-level globals, runs once)
# ---------------------------------------------------------------------------

# Ensure the SentenceTransformer model is loaded safely
if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
    sentence_model = SentenceTransformer(config.MODEL_NAME, use_auth_token=os.getenv("HF_TOKEN"))

spotify_df = pd.read_csv(config.DATASET_PATH)
try:
    tmdb_df = pd.read_csv(config.TMDB_PATH)
except FileNotFoundError:
    logging.error(f"TMDB file not found: {config.TMDB_PATH}")
    raise

# Preprocess TMDB: ensure year + build tone_text (tone-based indexing, not plot)
if "year" not in tmdb_df.columns:
    if "release_date" in tmdb_df.columns:
        tmdb_df["year"] = pd.to_datetime(tmdb_df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    else:
        tmdb_df["year"] = 0

tmdb_df["tone_text"] = tmdb_df.apply(build_tone_text, axis=1)

# Cache FAISS index per filter hash
INDEX_CACHE: dict[str, tuple] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_filters(filters: dict) -> str:
    payload = "|".join([f"{k}={filters.get(k)}" for k in sorted(filters.keys())])
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def get_best_audio_match(track: dict, spotify_df: pd.DataFrame) -> dict | None:
    """
    Match a Spotify track (name/artist) to a row in dataset.csv and return audio features dict.
    Copied/adapted from prior version (exact/partial/fuzzy).
    """
    track_name = str(track.get("name", "")).lower()
    artist_name = str(track.get("artist", "")).lower()
    if not track_name or not artist_name:
        return None

    if "track_name" not in spotify_df.columns or "artists" not in spotify_df.columns:
        return None

    # exact match
    exact_match = spotify_df[
        (spotify_df["track_name"].astype(str).str.lower() == track_name)
        & (spotify_df["artists"].astype(str).str.lower().str.contains(artist_name, na=False))
    ]
    if not exact_match.empty:
        return exact_match.iloc[0].to_dict()

    # partial match
    partial_match = spotify_df[
        (spotify_df["track_name"].astype(str).str.lower().str.contains(track_name, na=False))
        & (spotify_df["artists"].astype(str).str.lower().str.contains(artist_name, na=False))
    ]
    if not partial_match.empty:
        return partial_match.iloc[0].to_dict()

    # fuzzy match
    names = spotify_df["track_name"].astype(str).str.lower().tolist()
    artists = spotify_df["artists"].astype(str).str.lower().tolist()

    name_scores = process.cdist([track_name], names, scorer=fuzz.token_sort_ratio)[0]
    artist_scores = process.cdist([artist_name], artists, scorer=fuzz.token_set_ratio)[0]
    combined_scores = 0.6 * name_scores + 0.4 * artist_scores

    best_idx = int(np.argmax(combined_scores))
    if float(combined_scores[best_idx]) > 65:
        return spotify_df.iloc[best_idx].to_dict()
    return None


def _track_to_features(track: dict) -> dict:
    audio_features = get_best_audio_match(track, spotify_df) or {}
    descriptor, cinematic_vec = project_to_cinematic(audio_features)

    genre_text = f"{track.get('name','')} {track.get('artist','')} {' '.join(track.get('artist_genres', []))}"
    detected = detect_genres_from_text(genre_text)
    enriched_descriptor = enrich_with_genre(descriptor, detected)

    text_embedding = sentence_model.encode([enriched_descriptor], show_progress_bar=False)[0]

    return {
        "track": track,
        "audio_features": audio_features,
        "descriptor": enriched_descriptor,
        "cinematic_vector": cinematic_vec,
        "text_embedding": np.asarray(text_embedding, dtype="float32"),
    }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------


app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY or "change-me"


@app.get("/")
def index():
    authed = bool(session.get("token_info"))
    filters = session.get("filters") or {}
    return render_template("index.html", authed=authed, filters=filters, authed_flag=request.args.get("authed") == "1")


@app.get("/login")
def login():
    return redirect(get_auth_url())


@app.get("/callback")
def callback():
    code = request.args.get("code")
    if not code:
        return render_template("error.html", message="Authorization failed: missing code."), 400

    token_info = get_token_from_code(code)
    session["token_info"] = token_info
    return redirect(url_for("index", authed=1))


@app.post("/recommend")
def recommend():
    token_info = session.get("token_info")
    if not token_info:
        return redirect(url_for("login"))

    # refresh token if needed
    try:
        token_info = refresh_if_expired(token_info)
        session["token_info"] = token_info
    except Exception:
        pass

    # 2. filters
    filters = {
        "language": (request.form.get("language") or "").strip(),
        "genre": (request.form.get("genre") or "").strip(),
        "min_rating": (request.form.get("min_rating") or "").strip(),
        "min_popularity": (request.form.get("min_popularity") or "").strip(),
        "recent_only": request.form.get("recent_only") == "y",
    }
    session["filters"] = filters

    # Normalize numeric values
    min_rating = float(filters["min_rating"]) if filters["min_rating"] else None
    min_popularity = float(filters["min_popularity"]) if filters["min_popularity"] else None

    cache_key = _hash_filters(filters)
    if cache_key in INDEX_CACHE:
        faiss_index, movie_embeddings, filtered_tmdb = INDEX_CACHE[cache_key]
    else:
        filtered_tmdb = apply_filters(
            tmdb_df,
            language=filters["language"] or None,
            genre=filters["genre"] or None,
            min_rating=min_rating,
            min_popularity=min_popularity,
            recent_only=bool(filters["recent_only"]),
        )
        if filtered_tmdb.empty:
            return render_template("error.html", message="No movies match your filters. Try different settings."), 400

        faiss_index, movie_embeddings, filtered_tmdb = build_movie_index(filtered_tmdb, sentence_model)
        INDEX_CACHE[cache_key] = (faiss_index, movie_embeddings, filtered_tmdb)

    sp = get_spotify_client(token_info)
    tracks = get_top_tracks(sp, limit=config.TOP_TRACKS_LIMIT, time_range=config.TIME_RANGE)

    logging.debug(f"Tracks retrieved: {len(tracks)}")

    music_features = [_track_to_features(t) for t in tracks]

    logging.debug(f"Music features built: {len(music_features)}")

    recommender = MusicToMovieRecommender(filtered_tmdb, faiss_index, movie_embeddings, sentence_model)

    individual_recs = []
    for tf in music_features:
        recs = recommender.recommend_for_track(tf, top_k=5)
        logging.debug(f"Recommendations for track '{tf['track']['name']}': {recs}")
        individual_recs.append({"track": tf["track"], "descriptor": tf["descriptor"], "mood": None, "movies": recs})
        # More explicit debug: show types, counts and movie titles for easier triage
        try:
            titles = [m.get("title") for m in recs] if isinstance(recs, list) else []
        except Exception:
            titles = []
        logging.debug(f"Track features type: {type(tf)}, track key type: {type(tf.get('track'))}")
        logging.debug(f"Recommendations count for track '{tf.get('track', {}).get('name', '')}': {len(recs) if isinstance(recs, list) else 'N/A'}")
        logging.debug(f"Recommendation titles: {titles}")

    logging.debug(f"Length of individual_recs before render_template: {len(individual_recs)}")
    logging.debug(f"Final individual_recs: {len(individual_recs)}")

    profile_recs = recommender.recommend_for_profile(music_features, top_k=10)
    vibe = recommender.analyze_vibe(music_features)

    return render_template(
        "results.html",
        filters=filters,
        tracks=tracks,
        individual_recs=individual_recs,
        profile_recs=profile_recs,
        vibe=vibe,
    )


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1].split('=')[1]) if len(sys.argv) > 1 and '--port=' in sys.argv[1] else int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)  # Disabled debug mode to avoid reloader issues
