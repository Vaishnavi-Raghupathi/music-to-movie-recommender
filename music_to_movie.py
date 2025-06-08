import spotipy
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import asyncio
from openai import AsyncOpenAI
import warnings
import faiss
import hashlib
from rapidfuzz import process, fuzz
from difflib import SequenceMatcher
import time
from functools import lru_cache


warnings.filterwarnings('ignore')

# Load the environment 
load_dotenv()
perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
aclient = AsyncOpenAI(api_key=perplexity_api_key, base_url="https://api.perplexity.ai")
spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID")
spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")


sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


from spotipy.oauth2 import SpotifyOAuth
sp_oauth = SpotifyOAuth(
    client_id=spotify_client_id,
    client_secret=spotify_client_secret,
    redirect_uri='http://127.0.0.1:8888',
    scope='user-library-read user-top-read'
)

def load_datasets():
    # Spotify dataset
    spotify_dtypes = {
        'track_name': 'category',
        'artists': 'category',
        'danceability': 'float32',
        'energy': 'float32',
        'speechiness': 'float32',
        'acousticness': 'float32',
        'instrumentalness': 'float32',
        'liveness': 'float32',
        'valence': 'float32',
        'tempo': 'float32'
    }
    spotify_df = pd.read_csv('dataset.csv', dtype=spotify_dtypes, engine='pyarrow')

    # TMDB dataset
    tmdb_dtypes = {
        'title': 'category',
        'original_language': 'category',
        'genres': 'object',
        'keywords': 'object'
    }
    use_cols = ['title', 'vote_average', 'original_language', 'original_title',
                'overview', 'popularity', 'tagline', 'genres', 'production_countries', 'keywords', 'release_date']
    tmdb_df = pd.read_csv('TMDB_movie_dataset_v11.csv',
                         usecols=use_cols,
                         dtype=tmdb_dtypes,
                         engine='pyarrow')
    return spotify_df, tmdb_df

def initialize_spotify():
    """Authenticate user and return Spotify client."""
    auth_url = sp_oauth.get_authorize_url()
    print("Click to authorize:", auth_url)
    
    try:
        code = input(" Paste the FULL redirect URL: ").split("code=")[1]
    except IndexError:
        print(" Invalid URL. Retry.")
        exit()  
    
    token_info = sp_oauth.get_access_token(code)
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
    sp = spotipy.Spotify(auth=token_info['access_token'])
    return sp

def get_user_top_tracks(sp, limit=5, time_range='medium_term'):
    results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
    tracks_data = []
    for item in results['items']:
        tracks_data.append({
            'name': item['name'],
            'artist': item['artists'][0]['name'],
        })
    return tracks_data


def get_best_audio_match(track, spotify_df):
    track_name = track["name"].lower()
    artist_name = track["artist"].lower()

    #exact match
    exact_match = spotify_df[
        (spotify_df["track_name"].str.lower() == track_name) &
        (spotify_df["artists"].str.lower().str.contains(artist_name))
    ]
    if not exact_match.empty:
        return exact_match.iloc[0].to_dict()

    #partial match
    partial_match = spotify_df[
        (spotify_df["track_name"].str.lower().str.contains(track_name)) &
        (spotify_df["artists"].str.lower().str.contains(artist_name))
    ]
    if not partial_match.empty:
        return partial_match.iloc[0].to_dict()

    #vectorized fuzzy match
    names = spotify_df["track_name"].str.lower().tolist()
    artists = spotify_df["artists"].str.lower().tolist()

    name_scores = process.cdist([track_name], names, scorer=fuzz.token_sort_ratio)[0]
    artist_scores = process.cdist([artist_name], artists, scorer=fuzz.token_set_ratio)[0]
    combined_scores = 0.6 * name_scores + 0.4 * artist_scores

    best_idx = np.argmax(combined_scores)
    if combined_scores[best_idx] > 65:  # threshold can be tuned
        return spotify_df.iloc[best_idx].to_dict()

    #no match
    return None

async def get_keywords_batch(tracks, aclient):
    #batch process keywords for all tracks using your detailed prompt.
    tasks = []
    for track in tracks:
        prompt = (f"""
Analyze the song '{track['name']}' by '{track['artist']}' and return a **comma-separated list of 15 MAX keywords** that accurately describe:

1. The musical mood (e.g. melancholic, energetic, dreamy)
2. The emotional tone/themes (e.g. heartbreak, yearning, anxiety)
3. The musical style or genre (e.g. Tamil pop, indie, shoegaze)
4. The cinematic or visual aesthetic (e.g. slow-paced, vibrant, noir, sad-romantic, realistic)
5. The language and cultural vibe (e.g. Tamil, regional, nostalgic India)

These keywords will be used to match songs with movies that share similar moods or storytelling styles. Avoid repetition. Be precise. DO NOT give explanations.

Return ONLY a single comma-separated list of keywords.
""")
        messages = [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": prompt}
        ]
        await asyncio.sleep(2)
        tasks.append(
            aclient.chat.completions.create(
                model="sonar-pro",
                messages=messages,
                max_tokens=64,
                temperature=0.2,
            )
        )
    responses = await asyncio.gather(*tasks)
    return [r.choices[0].message.content.strip() for r in responses]

    
def preprocess_movie_data(tmdb_df):
    tmdb_df = tmdb_df.dropna(subset=['title', 'overview'])
    tmdb_df['tagline'] = tmdb_df['tagline'].fillna('')
    tmdb_df['genres'] = tmdb_df['genres'].astype(str).fillna('unknown')
    tmdb_df['keywords'] = tmdb_df['keywords'].astype(str).fillna('')
    tmdb_df['original_language'] = tmdb_df['original_language'].fillna('en')
    tmdb_df['year'] = pd.to_datetime(
        tmdb_df['release_date'], errors='coerce'
    ).dt.year.fillna(0).astype(int)

    # convert all to string before concatenation
    tmdb_df['title'] = tmdb_df['title'].astype(str)
    tmdb_df['overview'] = tmdb_df['overview'].astype(str)
    tmdb_df['tagline'] = tmdb_df['tagline'].astype(str)
    tmdb_df['genres'] = tmdb_df['genres'].astype(str)
    tmdb_df['keywords'] = tmdb_df['keywords'].astype(str)

    tmdb_df['combined_text'] = (
        tmdb_df['title'] + ' ' +
        tmdb_df['overview'] + ' ' +
        tmdb_df['tagline'] + ' ' +
        tmdb_df['genres'] + ' ' +
        tmdb_df['keywords']
    )
    return tmdb_df


def create_music_feature_vectors(tracks_data, spotify_df):
    """Create comprehensive feature vectors for music tracks"""
    music_features = []
    
    for track in tracks_data: 
        # get audio features
        audio_features = get_best_audio_match(track, spotify_df)
        
        # generate keywords
        keywords = track['keywords']
        
        # create combined text for embedding
        combined_text = f"{track['name']} {track['artist']} {keywords}"
        
        # get text embedding
        text_embedding = sentence_model.encode([combined_text])[0]
        
        # prepare numerical features
        if audio_features is not None:
            numerical_features = [
                audio_features.get('danceability', 0.55),
                audio_features.get('energy', 0.6),
                audio_features.get('speechiness', 0.05),
                audio_features.get('acousticness', 0.3),
                audio_features.get('instrumentalness', 0.1),
                audio_features.get('liveness', 0.2),
                audio_features.get('valence', 0.5),
                audio_features.get('tempo', 120) / 200,  # Normalize tempo
                track.get('popularity', 50) / 100,       # Normalize popularity
                track.get('duration_ms', 210000) / 300000, # Normalize duration
            ]
        else:
            numerical_features = [
                0.55,  # danceability
                0.6,   # energy
                0.05,  # speechiness
                0.3,   # acousticness
                0.1,   # instrumentalness
                0.2,   # liveness
                0.5,   # valence
                120 / 200,  # tempo normalized
                0.5,   # popularity normalized
                210000 / 300000  # duration normalized
            ]
      
        # combine features
        combined_features = np.concatenate([text_embedding, numerical_features])
        
        music_features.append({
            'track': track,
            'features': combined_features,
            'text_embedding': text_embedding,
            'numerical_features': numerical_features,
            'keywords': keywords
        })
    
    return music_features

def apply_movie_filters(tmdb_df, filters):
    """Apply user-specified filters to movie dataset"""
    # language filter
    if filters['language']:
        tmdb_df = tmdb_df[tmdb_df['original_language'] == filters['language']]
    
    # genre filter
    if filters['genre']:
        tmdb_df = tmdb_df[tmdb_df['genres'].str.contains(filters['genre'], case=False, na=False)]
    
    # rating filter
    if filters['min_rating']:
        try:
            tmdb_df = tmdb_df[tmdb_df['vote_average'] >= float(filters['min_rating'])]
        except ValueError:
            print("Invalid rating format, skipping rating filter")
    
    # popularity filter
    if filters['min_popularity']:
        try:
            tmdb_df = tmdb_df[tmdb_df['popularity'] >= float(filters['min_popularity'])]
        except ValueError:
            print("❗ Invalid popularity format, skipping popularity filter")
    
    # year filter
    if filters['year_range'] == 'y':
        tmdb_df = tmdb_df[tmdb_df['year'] >= 2015]
    
    return tmdb_df.reset_index(drop=True)



def create_movie_feature_vectors(tmdb_df):
    """Create feature vectors with FAISS index caching"""
    # generate unique cache key
    model_name = "all-MiniLM-L6-v2"
    data_hash = hashlib.md5(pd.util.hash_pandas_object(tmdb_df).values).hexdigest()
    cache_dir = "faiss_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    index_file = f"{cache_dir}/{model_name}_{data_hash}.index"
    embeddings_file = f"{cache_dir}/{model_name}_{data_hash}.npy"

    if os.path.exists(index_file) and os.path.exists(embeddings_file):
        print("Loading cached FAISS index and embeddings...")
        index = faiss.read_index(index_file)
        movie_embeddings = np.load(embeddings_file)
        return index, movie_embeddings

    print("Creating movie embeddings and FAISS index...")
    # generate embeddings
    movie_embeddings = sentence_model.encode(tmdb_df['combined_text'].tolist())
    
    # normalize embeddings for cosine similarity
    faiss.normalize_L2(movie_embeddings)
    
    # create and train FAISS index
    d = movie_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner product = cosine similarity
    index.add(movie_embeddings.astype('float32'))
    
    # save to cache
    faiss.write_index(index, index_file)
    np.save(embeddings_file, movie_embeddings)
    
    return index, movie_embeddings


class MusicToMovieRecommender:
    def __init__(self, tmdb_df, faiss_index, movie_embeddings):
        self.tmdb_df = tmdb_df
        self.index = faiss_index
        self.movie_embeddings = movie_embeddings

    def recommend_for_single_track(self, track_features, top_k=5):
        #FAISS-based recommendations
        # Get normalized track embedding
        track_embedding = track_features['text_embedding'].astype('float32')
        faiss.normalize_L2(track_embedding.reshape(1, -1))
        
        # search FAISS index
        distances, indices = self.index.search(track_embedding, top_k)
        
        return self._format_recommendations(indices[0], distances[0])

    def batch_recommend(self, music_features_list, top_k=5):
        #Batch FAISS recommendations
        # Extract and normalize all track embeddings
        track_embeddings = np.array([t['text_embedding'] for t in music_features_list], dtype='float32')
        faiss.normalize_L2(track_embeddings)
        
        # Search all at once
        distances, indices = self.index.search(track_embeddings, top_k)
        
        return [self._format_recommendations(batch_indices, batch_distances) 
                for batch_indices, batch_distances in zip(indices, distances)]

    def _format_recommendations(self, indices, distances):
        return [{
            'title': self.tmdb_df.iloc[idx]['title'],
            'overview': self.tmdb_df.iloc[idx]['overview'][:200] + '...',
            'genres': self.tmdb_df.iloc[idx]['genres'],
            'similarity_score': float(dist),
            'vote_average': self.tmdb_df.iloc[idx]['vote_average'],
            'original_language': self.tmdb_df.iloc[idx]['original_language']
        } for idx, dist in zip(indices, distances)]
    

    def recommend_for_profile(self, music_features_list, top_k=10):
        """Recommend movies based on overall music profile"""
        # Create weighted average of all track features
        all_features = np.array([track['features'] for track in music_features_list])
        
        # Weight by popularity or use equal weights
        weights = np.ones(len(all_features)) / len(all_features)
        profile_vector = np.average(all_features, axis=0, weights=weights).reshape(1, -1)
        
        # Extract text embedding portion (first 384 elements)
        text_embedding_dim = 384  # all-MiniLM-L6-v2 output size
        profile_vector_text = profile_vector[:, :text_embedding_dim].astype('float32')
        
        # Normalize and search
        faiss.normalize_L2(profile_vector_text)
        distances, indices = self.index.search(profile_vector_text, top_k)
        
        recommendations = []
        for idx, dist in zip(indices[0], distances[0]):
            movie = self.tmdb_df.iloc[idx]
            recommendations.append({
                'title': movie['title'],
                'overview': movie['overview'][:200] + '...',
                'genres': movie['genres'],
                'similarity_score': float(dist),
                'vote_average': movie['vote_average'],
                'original_language': movie['original_language']
            })
        
        return recommendations



    
    def analyze_music_vibe(self, music_features_list):
        """Analyze overall vibe of user's music profile"""
        # extract keywords from all tracks
        all_keywords = []
        for track in music_features_list:
            all_keywords.extend(track['keywords'].split(', '))
        
        # Count keyword frequency
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Analyze numerical features
        numerical_data = np.array([track['numerical_features'] for track in music_features_list])
        avg_features = np.mean(numerical_data, axis=0)
        
        feature_names = ['danceability', 'energy', 'speechiness', 'acousticness', 
                        'instrumentalness', 'liveness', 'valence', 'tempo', 
                        'popularity', 'duration']
        
        vibe_analysis = {
            'top_keywords': top_keywords,
            'audio_profile': {name: round(val, 3) for name, val in zip(feature_names, avg_features)},
            'dominant_mood': self._determine_mood(avg_features)
        }
        
        return vibe_analysis
    
    def _determine_mood(self, avg_features):
        """Determine dominant mood from audio features"""
        valence, energy, danceability = avg_features[6], avg_features[1], avg_features[0]
        
        if valence > 0.7 and energy > 0.7:
            return "Energetic and Positive"
        elif valence > 0.7 and energy < 0.4:
            return "Calm and Happy"
        elif valence < 0.3 and energy > 0.6:
            return "Aggressive and Intense"
        elif valence < 0.3 and energy < 0.4:
            return "Melancholic and Introspective"
        elif danceability > 0.7:
            return "Rhythmic and Groovy"
        else:
            return "Balanced and Moderate"
    
    @staticmethod
    def get_user_filters():
        #collect user preferences for movie filtering
        print("\nCustomize Your Movie Preferences:")
        
        filters = {
            'language': input("Preferred language code (e.g., en, ta, hi) [Enter to skip]: ").lower().strip(),
            'genre': input("Preferred genre (e.g., Action, Romance) [Enter to skip]: ").title().strip(),
            'min_rating': input("Minimum rating (1-10) [Enter to skip]: ").strip(),
            'min_popularity': input("Minimum popularity score [Enter to skip]: ").strip(),
            'year_range': input("Only recent movies? (2015+) [y/N]: ").lower().strip()
        }
        
        return filters          
    

def main():
    print("Initializing Music-to-Movie Recommendation System...")

    # load datasetst
    print(" Loading datasets...")
    spotify_df, tmdb_df = load_datasets()
    tmdb_df = preprocess_movie_data(tmdb_df)

    filters = MusicToMovieRecommender.get_user_filters()
    filtered_tmdb = apply_movie_filters(tmdb_df, filters)
    print(f"\nFiltered to {len(filtered_tmdb)} movies based on your preferences")
    filtered_tmdb.to_csv('preprocessed_movies.csv', index=False)

    
    sp = initialize_spotify()
    tracks_data = get_user_top_tracks(sp)

    print("\n Generating keywords via Perplexity...")
    loop = asyncio.get_event_loop()
    all_keywords = loop.run_until_complete(get_keywords_batch(tracks_data, aclient))
    
    # attach keywords to tracks
    for i, track in enumerate(tracks_data):
        track['keywords'] = all_keywords[i]

    print(" Processing user's music profile...")

    music_features = create_music_feature_vectors(tracks_data, spotify_df)
    faiss_index, movie_embeddings = create_movie_feature_vectors(filtered_tmdb)
    faiss.write_index(faiss_index, 'faiss_cache/recommender.index')
    recommender = MusicToMovieRecommender(filtered_tmdb, faiss_index, movie_embeddings)
    
    print("\n" + "="*80)
    print("INDIVIDUAL TRACK RECOMMENDATIONS")
    print("="*80)
    
    # individual track recommendations
    # batch recommendations for all tracks
    all_recommendations = recommender.batch_recommend(music_features, top_k=3)
    for i, recs in enumerate(all_recommendations):
        track = music_features[i]['track']
        print(f"\n Track: '{track['name']}' by {track['artist']}")
        print(f"Keywords: {music_features[i]['keywords']}")
        print(" Recommended Movies:")
        for j, rec in enumerate(recs, 1):
            print(f"  {j}. {rec['title']} ({rec['original_language']}) - Score: {rec['similarity_score']:.3f}")
            print(f"     Genres: {rec['genres']}")
            print(f"     Overview: {rec['overview']}")
            print()

    
    # overall profile analysis
    print("\n" + "="*80)
    print("OVERALL MUSIC VIBE ANALYSIS")
    print("="*80)
    
    vibe_analysis = recommender.analyze_music_vibe(music_features)
    
    print("Your Music Vibe Profile:")
    print(f"Dominant Mood: {vibe_analysis['dominant_mood']}")
    print("\nTop Musical Themes:")
    for keyword, count in vibe_analysis['top_keywords'][:5]:
        print(f"  • {keyword} (appears {count} times)")
    
    print("\nAudio Feature Profile:")
    for feature, value in vibe_analysis['audio_profile'].items():
        print(f"  • {feature}: {value}")
    
    print("\n" + "="*80)
    print("OVERALL PROFILE RECOMMENDATIONS")
    print("="*80)
    
    profile_recommendations = recommender.recommend_for_profile(music_features, top_k=8)
    
    print("Movies matching your overall music vibe:")
    for i, rec in enumerate(profile_recommendations, 1):
        print(f"{i}. {rec['title']} ({rec['original_language']}) - Score: {rec['similarity_score']:.3f}")
        print(f"   Rating: {rec['vote_average']}/10 | Genres: {rec['genres']}")
        print(f"   {rec['overview']}")
        print()

# execute the recommendation system
if __name__ == "__main__":
    main()