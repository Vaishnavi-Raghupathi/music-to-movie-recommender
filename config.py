import os

from dotenv import load_dotenv

load_dotenv()

# Load from .env
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

# Constants
MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_CACHE_DIR = "faiss_cache"
DATASET_PATH = "dataset.csv"
TMDB_PATH = "TMDB_movie_dataset_v11.csv"
CINEMATIC_VECTOR_DIM = 6
TEXT_EMBEDDING_DIM = 384
TOP_TRACKS_LIMIT = 10
TIME_RANGE = "medium_term"

