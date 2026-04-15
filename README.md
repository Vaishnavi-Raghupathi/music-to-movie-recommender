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

1. Push repo to GitHub (DO NOT commit .env, datasets, or faiss_cache/)
2. Connect repo to Railway
3. Add env vars in Railway dashboard
4. Upload dataset.csv and TMDB_movie_dataset_v11.csv via Railway volume 
   or include them in the repo if size allows (<100MB)

### Spotify App Settings

Add your production URL to Redirect URIs in Spotify Developer Dashboard:

```
https://your-app.railway.app/callback
```

### Notes

- `faiss_cache/` is gitignored and will be rebuilt on first request per filter combination. First load will be slow (~30s). Subsequent loads use cache.

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

## New Features

### Ask a Question

- **Description**: Users can now ask questions about the research paper text directly in the app.
- **How It Works**:
  1. Enter your question in the "Ask a Question" text input.
  2. Click the "Get Answer" button.
  3. The app uses the `ask_question` function to generate a concise, technical answer based on the paper text.

### Improved Debugging and Stability

- **Safe Model Loading**: The SentenceTransformer model is now loaded safely to avoid issues with the Flask reloader.
- **Default Port**: The app defaults to port 5001 to prevent conflicts with other applications.
- **Enhanced Debug Logs**: Added detailed debug logs to trace the recommendation pipeline, including `tracks`, `music_features`, and `individual_recs`.

---

## License

MIT
