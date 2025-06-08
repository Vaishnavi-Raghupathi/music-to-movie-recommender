from flask import Flask, redirect, request, session, render_template_string
from flask_session import Session
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from spotipy.cache_handler import CacheFileHandler
import numpy as np
import faiss
import asyncio
from openai import AsyncOpenAI

from music_to_movie import (
    MusicToMovieRecommender,
    load_datasets,
    preprocess_movie_data,
    apply_movie_filters,
    create_music_feature_vectors,
    get_keywords_batch,
    create_movie_feature_vectors
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key")
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

def session_cache_path():
    return f".cache-{session.sid}"

aclient = AsyncOpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

spotify_df, tmdb_df = load_datasets()
tmdb_df = preprocess_movie_data(tmdb_df)

def get_recommender():
    filters = session.get('filters', {
        'language': '', 
        'genre': '',
        'min_rating': '',
        'min_popularity': '',
        'year_range': ''
    })
    filtered_tmdb = apply_movie_filters(tmdb_df, filters)
    if filtered_tmdb.empty:
        raise ValueError("No movies match your filters. Please try different settings.")
    faiss_index, movie_embeddings = create_movie_feature_vectors(filtered_tmdb)
    return MusicToMovieRecommender(filtered_tmdb, faiss_index, movie_embeddings)

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>MusicMovieMatch.exe</title>
        <link rel="stylesheet" href="https://unpkg.com/98.css@0.1.22/dist/98.css">
        <style>
            body { background: #FFFFE0; }
            .centered { display: flex; justify-content: center; align-items: center; height: 100vh; }
        </style>
    </head>
    <body>
    <div class="centered">
      <div class="window" style="width: 400px;">
        <div class="title-bar">
          <div class="title-bar-text">MusicMovieMatch.exe</div>
          <div class="title-bar-controls">
            <button aria-label="Minimize"></button>
            <button aria-label="Maximize"></button>
            <button aria-label="Close"></button>
          </div>
        </div>
        <div class="window-body">
          <h2>Welcome to MusicMovieMatch</h2>
          <p>Go to <a href="/filters">Set Preferences and Start</a></p>
        </div>
      </div>
    </div>
    </body>
    </html>
    ''')

@app.route('/filters', methods=['GET', 'POST'])
def filter_form():
    if request.method == 'POST':
        session['filters'] = {
            'language': request.form.get('language', ''),
            'genre': request.form.get('genre', ''),
            'min_rating': request.form.get('min_rating', ''),
            'min_popularity': request.form.get('min_popularity', ''),
            'year_range': 'y' if request.form.get('year_range') == 'y' else 'n'
        }
        return redirect('/login')
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Set Preferences - MusicMovieMatch.exe</title>
        <link rel="stylesheet" href="https://unpkg.com/98.css@0.1.22/dist/98.css">
        <style>
            body { background: #FFFFE0; }
            .centered { display: flex; justify-content: center; align-items: center; height: 100vh; }
            .window { min-width: 350px; }
            .window-body { padding: 20px 24px; }
            fieldset { margin-bottom: 12px; }
            label { display: block; margin-bottom: 8px; }
            input[type="text"], select { width: 100%; margin-bottom: 12px; }
            .field-row-stacked { margin-bottom: 12px; }
            button { margin-top: 8px; }
        </style>
    </head>
    <body>
    <div class="centered">
      <div class="window" style="width: 350px;">
        <div class="title-bar">
          <div class="title-bar-text">Movie Preferences</div>
          <div class="title-bar-controls">
            <button aria-label="Minimize"></button>
            <button aria-label="Maximize"></button>
            <button aria-label="Close"></button>
          </div>
        </div>
        <div class="window-body">
          <form method="post">
            <fieldset>
              <legend>Set your movie filters</legend>
              <div class="field-row-stacked">
                <label for="language">Language</label>
                <input type="text" id="language" name="language" placeholder="e.g. en, ta, hi">
              </div>
              <div class="field-row-stacked">
                <label for="genre">Genre</label>
                <input type="text" id="genre" name="genre" placeholder="e.g. Action, Romance">
              </div>
              <div class="field-row-stacked">
                <label for="min_rating">Min Rating</label>
                <input type="text" id="min_rating" name="min_rating" placeholder="1-10">
              </div>
              <div class="field-row-stacked">
                <label for="min_popularity">Min Popularity</label>
                <input type="text" id="min_popularity" name="min_popularity" placeholder="e.g. 20">
              </div>
              <div class="field-row-stacked">
                <label><input type="checkbox" name="year_range" value="y"> Only recent movies (2015+)</label>
              </div>
              <div class="field-row-stacked" style="text-align:center;">
                <button type="submit">Continue to Spotify Login</button>
              </div>
            </fieldset>
          </form>
        </div>
      </div>
    </div>
    </body>
    </html>
    ''')

@app.route('/login', methods=['GET', 'POST'])  # Allow both GET and POST
def login():
    sp_oauth = SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
        scope='user-library-read user-top-read',
        cache_handler=CacheFileHandler(session_cache_path())
    )
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    sp_oauth = SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
        scope='user-library-read user-top-read',
        cache_handler=CacheFileHandler(session_cache_path())
    )
    code = request.args.get('code')
    if not code:
        return render_template_string('''
        <div class="window" style="width:400px; margin:50px auto;">
          <div class="title-bar"><div class="title-bar-text">Error</div></div>
          <div class="window-body">Authorization failed: No code received</div>
        </div>
        ''')
    try:
        token_info = sp_oauth.get_access_token(code)
        session['token_info'] = token_info
        return redirect('/process')
    except Exception as e:
        return render_template_string(f'''
        <div class="window" style="width:400px; margin:50px auto;">
          <div class="title-bar"><div class="title-bar-text">Error</div></div>
          <div class="window-body">Authorization failed: {str(e)}</div>
        </div>
        ''')

@app.route('/process')
def process_recommendations():
    token_info = session.get('token_info')
    if not token_info:
        return redirect('/login')
    try:
        sp = spotipy.Spotify(auth=token_info['access_token'])
        tracks_data = []
        results = sp.current_user_top_tracks(limit=5)
        for item in results['items']:
            tracks_data.append({
                'name': item['name'],
                'artist': item['artists'][0]['name'],
                'id': item['id']
            })
        tracks_for_keywords = [{'name': t['name'], 'artist': t['artist']} for t in tracks_data]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        all_keywords = loop.run_until_complete(get_keywords_batch(tracks_for_keywords, aclient))
        for i, track in enumerate(tracks_data):
            track['keywords'] = all_keywords[i]
        music_features = create_music_feature_vectors(tracks_data, spotify_df)
        recommender = get_recommender()
        individual_recommendations = recommender.batch_recommend(music_features, top_k=3)
        profile_recommendations = recommender.recommend_for_profile(music_features, top_k=8)
        vibe_analysis = recommender.analyze_music_vibe(music_features)
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Results - MusicMovieMatch.exe</title>
            <link rel="stylesheet" href="https://unpkg.com/98.css@0.1.22/dist/98.css">
            <style>
                body { background: #FFFFE0; }
                .window-body { max-height: 80vh; overflow-y: auto; }
                .movie-card { margin-bottom: 12px; padding: 8px; border: 1px solid #bbb; border-radius: 4px; background: #fff; }
                .track-group { margin-bottom: 24px; }
            </style>
        </head>
        <body>
        <div class="window" style="width: 700px; margin: 40px auto;">
          <div class="title-bar">
            <div class="title-bar-text">MusicMovieMatch.exe - Results</div>
            <div class="title-bar-controls">
              <button aria-label="Minimize"></button>
              <button aria-label="Maximize"></button>
              <button aria-label="Close"></button>
            </div>
          </div>
          <div class="window-body">
            <div style="text-align:right;">
                <a href="/logout">Log Out</a>
            </div>                         
            <h1>Your Recommendations</h1>
            <h2>Track-Specific Matches</h2>
            {% for track_recs in individual_recs %}
            <fieldset class="track-group">
                <legend>Based on: {{ track_recs.track.name }} by {{ track_recs.track.artist }}</legend>
                {% for movie in track_recs.movies %}
                <div class="movie-card">
                    <b>{{ movie.title }}</b> <span style="color: #888;">({{ movie.similarity_score }})</span><br>
                    <span style="font-size: 90%;">{{ movie.overview }}</span>
                </div>
                {% endfor %}
            </fieldset>
            {% endfor %}
            <h2>Your Music Vibe Profile</h2>
            <fieldset>
                <legend>Vibe Analysis</legend>
                <b>Dominant Mood:</b> {{ vibe.dominant_mood }}<br>
                <b>Top Themes:</b>
                <ul>
                    {% for keyword, count in vibe.top_keywords %}
                    <li>{{ keyword }} ({{ count }} mentions)</li>
                    {% endfor %}
                </ul>
                <b>Audio Features:</b>
                <ul>
                    {% for feature, value in vibe.audio_profile.items() %}
                    <li>{{ feature }}: {{ "%.2f"|format(value) }}</li>
                    {% endfor %}
                </ul>
            </fieldset>
            <h2>Overall Profile Matches</h2>
            <p>Based on your complete music profile featuring:
                {% for track in profile_tracks %}{{ track }}{% if not loop.last %}, {% endif %}{% endfor %}
            </p>
            {% for movie in profile_recs %}
            <div class="movie-card">
                <b>{{ movie.title }}</b> <span style="color: #888;">({{ movie.similarity_score }})</span><br>
                <span style="font-size: 90%;">{{ movie.overview }}</span>
            </div>
            {% endfor %}
            <div style="margin-top:30px; text-align:center;">
                <button onclick="window.location.href='/'">Start Over</button>
            </div>
          </div>
        </div>
        </body>
        </html>
        ''', 
        individual_recs=[{
            'track': track['track'], 
            'movies': recs
        } for track, recs in zip(music_features, individual_recommendations)],
        profile_recs=profile_recommendations,
        profile_tracks=[t['track']['name'] for t in music_features],
        vibe=vibe_analysis)
    except Exception as e:
        return render_template_string(f'''
        <div class="window" style="width:500px; margin:50px auto;">
          <div class="title-bar"><div class="title-bar-text">Error</div></div>
          <div class="window-body">
            <div style="color:red; padding:20px; border:1px solid red; margin:20px;">
                Error generating recommendations: {str(e)}
            </div>
            <a href="/">Try Again</a>
          </div>
        </div>
        ''')
    
@app.route('/logout')
def logout():
    session.clear()
    cache_path = f".cache-{session.sid}"
    if os.path.exists(cache_path):
        os.remove(cache_path)
    return redirect('/')
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
