from __future__ import annotations

from rapidfuzz import fuzz


GENRE_CINEMA_MAP = {
    "shoegaze": "dreamlike hazy atmospheric introspective ethereal slow-burn wall-of-sound",
    "dream pop": "romantic melancholic tender floating soft-focus pastel",
    "post-rock": "epic slow-build emotional cathartic cinematic wordless transcendent",
    "indie": "authentic understated naturalistic coming-of-age human",
    "uk indie": "bittersweet observational dry witty working-class northern",
    "indie folk": "melancholic intimate acoustic storytelling rural solitary",
    "math rock": "complex rhythmic precise angular cerebral restless",
    "ambient": "meditative atmospheric immersive sparse slow dissolving",
    "indie metal": "tension catharsis dark heavy psychological confrontational",
    "noise rock": "abrasive chaotic intense dissonant raw confrontational",
    "emo": "raw vulnerable emotional confessional dramatic teenage",
    "post-punk": "angular cold detached urban existential alienated",
    "darkwave": "gothic haunting cold atmospheric nocturnal",
    "lo-fi": "nostalgic hazy intimate low-key understated bedroom",
    "art rock": "experimental unconventional cerebral theatrical",
    "chamber pop": "orchestral lush literary melancholic refined",
    "slowcore": "desolate minimal sparse aching quiet devastation",
    "midwest emo": "earnest vulnerable confessional suburban lonely",
    "black metal": "bleak vast cold nihilistic raw elemental",
    "shoegaze metal": "crushing atmospheric dense wall-of-sound heavy dreamlike",
    "folk": "storytelling rural nostalgic earthy honest",
    "singer-songwriter": "confessional intimate acoustic personal raw",
}


def _norm(s: str) -> str:
    return " ".join((s or "").lower().strip().split())


def detect_genres_from_text(text: str) -> list[str]:
    """
    Fuzzy match input text (artist + track combined) against GENRE_CINEMA_MAP keys.
    Returns all keys with score > 70.
    """
    hay = _norm(text)
    if not hay:
        return []

    detected: list[str] = []
    for key in GENRE_CINEMA_MAP.keys():
        score = fuzz.partial_ratio(hay, _norm(key))
        if score > 70:
            detected.append(key)
    return detected


def enrich_with_genre(base_descriptor: str, detected_genres: list[str]) -> str:
    """
    Appends cinema descriptors for each detected genre, then deduplicates words.
    """
    chunks = [base_descriptor or ""]
    for g in detected_genres or []:
        mapped = GENRE_CINEMA_MAP.get(g)
        if mapped:
            chunks.append(mapped)

    words: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for w in (chunk or "").split():
            lw = w.lower()
            if lw and lw not in seen:
                seen.add(lw)
                words.append(w)
    return " ".join(words).strip()


__all__ = ["GENRE_CINEMA_MAP", "detect_genres_from_text", "enrich_with_genre"]

