from app import app
from flask import url_for

data = {
    'vibe': {
        'dominant_mood': 'Happy',
        'top_descriptors': ['upbeat', 'pop'],
        'audio_averages': {'danceability': 0.8, 'energy': 0.9, 'valence': 0.7, 'tempo': 120.0}
    },
    'individual_recs': [
        {
            'track': {'name': 'Song 1', 'artist': 'Artist 1'},
            'mood': 'Happy',
            'movies': [
                {'title': 'Movie 1', 'original_language': 'en', 'vote_average': 8.5, 'genres': 'Comedy', 'overview': 'Funny movie.', 'similarity_score': 0.85}
            ]
        }
    ],
    'profile_recs': [
        {'title': 'Movie 2', 'original_language': 'en', 'vote_average': 7.5, 'genres': 'Drama', 'overview': 'Sad movie.', 'similarity_score': 0.75}
    ],
    'filters': {'language': 'en', 'genre': 'Comedy', 'min_rating': '7.0', 'min_popularity': '50', 'recent_only': True}
}

with app.test_request_context():
    from flask import render_template
    try:
        rendered = render_template('results.html', **data)
        print("Render successful!")
    except Exception as e:
        import traceback
        traceback.print_exc()
