from __future__ import annotations

from functools import lru_cache
from typing import Any

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI


_SCOPE = "user-top-read user-library-read"

def _get_oauth() -> SpotifyOAuth:
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI:
        raise RuntimeError(
            "Missing Spotify config. Ensure SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, "
            "and SPOTIFY_REDIRECT_URI are set in your environment/.env."
        )
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=_SCOPE,
    )


def get_auth_url() -> str:
    sp_oauth = _get_oauth()
    return sp_oauth.get_authorize_url()


def get_token_from_code(code: str) -> dict:
    sp_oauth = _get_oauth()
    token_info = sp_oauth.get_access_token(code, check_cache=False)
    return token_info


def refresh_if_expired(token_info: dict) -> dict:
    sp_oauth = _get_oauth()
    if sp_oauth.is_token_expired(token_info):
        return sp_oauth.refresh_access_token(token_info["refresh_token"])
    return token_info


def get_spotify_client(token_info: dict) -> spotipy.Spotify:
    return spotipy.Spotify(auth=token_info["access_token"])


@lru_cache(maxsize=256)
def _artist_genres(sp_access_token: str, artist_id: str) -> tuple[str, ...]:
    sp = spotipy.Spotify(auth=sp_access_token)
    data: dict[str, Any] = sp.artist(artist_id)
    genres = data.get("genres") or []
    if not isinstance(genres, list):
        return tuple()
    return tuple(str(g) for g in genres if g)


def get_top_tracks(sp: spotipy.Spotify, limit: int = 10, time_range: str = "medium_term") -> list[dict]:
    results: dict[str, Any] = sp.current_user_top_tracks(limit=int(limit), time_range=time_range)
    items = results.get("items") or []

    # Best-effort access token for artist genre calls.
    sp_access_token = getattr(getattr(sp, "auth_manager", None), "token", None) or getattr(sp, "_auth", None) or ""

    tracks: list[dict] = []
    for t in items:
        artists = t.get("artists") or []
        first_artist = artists[0] if artists else {}
        artist_name = str(first_artist.get("name", ""))
        artist_id = str(first_artist.get("id", ""))

        artist_genres: list[str] = []
        if sp_access_token and artist_id:
            try:
                artist_genres = list(_artist_genres(sp_access_token, artist_id))
            except Exception:
                artist_genres = []

        tracks.append(
            {
                "name": str(t.get("name", "")),
                "artist": artist_name,
                "popularity": int(t.get("popularity", 0) or 0),
                "duration_ms": int(t.get("duration_ms", 0) or 0),
                "artist_genres": artist_genres,
            }
        )
    return tracks


__all__ = [
    "get_auth_url",
    "get_token_from_code",
    "refresh_if_expired",
    "get_spotify_client",
    "get_top_tracks",
]

