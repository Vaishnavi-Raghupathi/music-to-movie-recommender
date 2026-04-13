# 🎵 MusicMatch 98

> **Music-to-Movie Recommender** with a Windows 95/98 aesthetic.  
> Analyses your Spotify listening history using audio feature analysis and recommends movies that match your vibe — no AI, no external APIs.

---

## How It Works

1. Log in with Spotify (OAuth2).
2. MusicMatch 98 fetches your top tracks and matches each one to audio features (danceability, energy, valence, tempo, etc.).
3. Each track is projected into a "cinematic descriptor" and matched against a TMDB movie dataset via semantic similarity.
4. You get per-track movie picks **and** an overall vibe profile.

---

## Local Development

```bash
# 1. Clone
git clone https://github.com/your-username/music-to-movie-recommender.git
cd music-to-movie-recommender

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and fill in env vars
cp .env.example .env
# Edit .env with your Spotify credentials

# 5. Place datasets in project root:
#    dataset.csv                  (Spotify audio features)
#    TMDB_movie_dataset_v11.csv   (TMDB movie metadata)

# 6. Run
flask run
# or
python app.py
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `SPOTIFY_CLIENT_ID` | From Spotify Developer Dashboard |
| `SPOTIFY_CLIENT_SECRET` | From Spotify Developer Dashboard |
| `SPOTIFY_REDIRECT_URI` | Must match exactly in Spotify app settings |
| `FLASK_SECRET_KEY` | Any long random string |
| `FLASK_DEBUG` | Set to `1` for debug mode (local only) |

---

## Deployment

### Railway

1. Push the repo to GitHub — **do NOT commit** `.env`, datasets (`dataset.csv`, `TMDB_movie_dataset_v11.csv`), or `faiss_cache/` (all are gitignored).
2. Go to [Railway](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Select this repository.
4. Add the following environment variables in the Railway dashboard:
   - `SPOTIFY_CLIENT_ID`
   - `SPOTIFY_CLIENT_SECRET`
   - `SPOTIFY_REDIRECT_URI` → `https://your-app.railway.app/callback`
   - `FLASK_SECRET_KEY`
5. Upload `dataset.csv` and `TMDB_movie_dataset_v11.csv` via a **Railway Volume** mounted at `/app`, or include them directly in the repo if combined size is under 100 MB.
6. Railway will use `railway.toml` and `Procfile` automatically — no extra configuration needed.

### Render

1. Push the repo to GitHub (same gitignore rules as above).
2. Go to [Render](https://render.com) → **New Web Service** → connect your repo.
3. Render will detect `render.yaml` and pre-fill build/start commands.
4. Set environment variables in the Render dashboard (marked `sync: false` in `render.yaml` so they are never committed to source).
5. Upload datasets via a Render **Disk** or bundle them in the repo if size allows.

### Spotify App Settings

In the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard), add your production callback URL to **Redirect URIs**:

```
https://your-app.railway.app/callback
# or
https://your-app.onrender.com/callback
```

---

## Performance Notes

> [!NOTE]
> `faiss_cache/` is gitignored and **rebuilt on demand** on first request per unique filter combination. The **first load after deploy will take ~30 seconds** while the FAISS index is constructed and embeddings are computed.  
> Subsequent requests with the same filters use the in-memory cache and respond in ~1–3 seconds.

---

## Project Structure

```
music-to-movie-recommender/
├── app.py                  # Flask application
├── config.py               # Configuration loader
├── core/
│   ├── audio_projection.py # Audio → cinematic vector projection
│   ├── genre_bridge.py     # Genre detection & enrichment
│   ├── movie_index.py      # FAISS index builder + filters
│   ├── recommender.py      # Recommendation engine
│   └── spotify_client.py   # Spotify OAuth + API helpers
├── templates/
│   ├── base.html           # Win98 base layout (98.css + taskbar)
│   ├── index.html          # Home / preference form
│   ├── results.html        # Vibe profile + movie recommendations
│   └── error.html          # Error dialog
├── static/
│   └── css/custom.css      # Win98 custom styles
├── Procfile                # Heroku / Railway process file
├── railway.toml            # Railway-specific config
├── render.yaml             # Render deployment manifest
└── requirements.txt
```

---

## License

MIT
