**Music-to-Movie Recommender**

Music-to-Movie Recommender is a machine learning application that recommends movies based on your Spotify music taste. By analyzing your top Spotify tracks, their audio features, and semantic keywords, the app matches your musical vibe to a vast movie database—delivering both track-specific and overall profile-wide movie suggestions.

---

**Features**

* Spotify OAuth Integration: Secure and seamless login to analyze your Spotify listening profile.
* Movie Filtering: Customize recommendations by language, genre, rating, popularity, and recency.
* Semantic Analysis: Leverages large language models and sentence transformers to extract meaningful keywords and vibes from your music.
* Audio Feature Matching: Aligns Spotify track audio features with movie metadata for accurate pairing.
* Personalized Recommendations: Unique movie suggestions per track and for your overall music taste.
* Multi-User Support: Individualized recommendations without data crossover.

---

**Demo**

1. Set your movie preferences.
2. Log in with Spotify.
3. View your music vibe analysis and personalized movie recommendations.

---

**Installation**

Clone the repository:
`git clone https://github.com/yourusername/music-to-movie-recommender.git`
`cd music-to-movie-recommender`

Install required dependencies:
`pip install -r requirements.txt`

Configure environment variables in a `.env` file:

```
SPOTIFY_CLIENT_ID=your_spotify_client_id  
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret  
SPOTIFY_REDIRECT_URI=http://localhost:5000/callback  
PERPLEXITY_API_KEY=your_perplexity_api_key  
FLASK_SECRET_KEY=your_flask_secret_key  
```

Add your datasets to the project folder:

* `dataset.csv` (Spotify audio features)
* `TMDB_movie_dataset_v11.csv` (Movie metadata)

Run the application:
`python app.py`

Open your browser and visit `http://localhost:5000`

---

**Tech Stack**

Backend: Flask, Spotipy, FAISS, Pandas, NumPy, Sentence Transformers
Frontend: 98.css, HTML, Jinja2
APIs: Spotify Web API, OpenAI / Perplexity API

---

**Contributing**

Pull requests and issues are welcome. For major changes, please open an issue first to discuss your ideas.

---

**License**

This project is licensed under the MIT License.

---

**Acknowledgements**

* Spotify Web API
* The Movie Database (TMDb)
* Sentence Transformers
* FAISS

Enjoy discovering movies inspired by your music!

---
