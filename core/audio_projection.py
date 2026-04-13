from __future__ import annotations

import numpy as np


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def project_to_cinematic(audio_features: dict) -> tuple[str, np.ndarray]:
    """
    Maps Spotify-style audio features -> (cinematic descriptor string, 6D cinematic vector).
    Pure math; no LLMs or external APIs.
    """
    # Step 1: extract + normalize
    danceability = _clamp01(float(audio_features.get("danceability", 0.0)))
    energy = _clamp01(float(audio_features.get("energy", 0.0)))
    speechiness = _clamp01(float(audio_features.get("speechiness", 0.0)))
    acousticness = _clamp01(float(audio_features.get("acousticness", 0.0)))
    instrumentalness = _clamp01(float(audio_features.get("instrumentalness", 0.0)))
    liveness = _clamp01(float(audio_features.get("liveness", 0.0)))
    valence = _clamp01(float(audio_features.get("valence", 0.0)))
    tempo_norm = _clamp01(float(audio_features.get("tempo", 0.0)) / 200.0)

    # Step 2: compute cinematic dims (clamp to 0-1)
    emotional_tone = _clamp01(valence * 0.6 + energy * (-0.2) + acousticness * 0.2 + 0.2)
    pacing = _clamp01(energy * 0.5 + danceability * 0.3 + tempo_norm * 0.2)
    texture = _clamp01(acousticness * 0.5 + instrumentalness * 0.3 + (1 - energy) * 0.2)
    intensity = _clamp01(energy * 0.4 + (1 - valence) * 0.3 + speechiness * 0.3)
    intimacy = _clamp01(acousticness * 0.4 + (1 - liveness) * 0.3 + (1 - energy) * 0.3)
    darkness = _clamp01((1 - valence) * 0.5 + energy * 0.3 + (1 - acousticness) * 0.2)

    vec = np.array(
        [emotional_tone, pacing, texture, intensity, intimacy, darkness],
        dtype="float32",
    )

    # Step 3: descriptors
    words: list[str] = []

    # emotional_tone
    if emotional_tone > 0.7:
        words.append("uplifting warm radiant hopeful")
    elif 0.4 <= emotional_tone <= 0.7:
        words.append("bittersweet wistful tender ambivalent")
    else:
        words.append("melancholic bleak desolate sorrowful")

    # pacing
    if pacing > 0.7:
        words.append("kinetic propulsive urgent frenetic")
    elif 0.4 <= pacing <= 0.7:
        words.append("measured deliberate steady")
    else:
        words.append("slow-burn meditative languid drifting")

    # texture
    if texture > 0.6:
        words.append("intimate sparse acoustic minimal")
    elif 0.3 <= texture <= 0.6:
        words.append("layered textured nuanced")
    else:
        words.append("dense epic orchestral produced")

    # intensity
    if intensity > 0.7:
        words.append("tense overwhelming visceral")
    elif 0.4 <= intensity <= 0.7:
        words.append("focused gripping absorbing")
    else:
        words.append("gentle subdued restrained quiet")

    # intimacy
    if intimacy > 0.6:
        words.append("personal confessional close observational")
    elif 0.3 <= intimacy <= 0.6:
        words.append("human grounded")
    else:
        words.append("grand sweeping cinematic distant")

    # darkness
    if darkness > 0.7:
        words.append("dark brooding noir shadowy")
    elif 0.4 <= darkness <= 0.7:
        words.append("complex morally-ambiguous")
    else:
        words.append("bright hopeful light earnest")

    # Step 4: return
    descriptor_string = " ".join(words)
    return descriptor_string, vec


def batch_project(tracks: list[dict]) -> list[tuple[str, np.ndarray]]:
    """
    Accepts a list of track dicts. Each track may either:
    - be an audio_features dict itself, or
    - contain an 'audio_features' key.
    """
    out: list[tuple[str, np.ndarray]] = []
    for t in tracks:
        feats = t.get("audio_features") if isinstance(t, dict) else None
        if feats is None and isinstance(t, dict):
            feats = t
        if not isinstance(feats, dict):
            feats = {}
        out.append(project_to_cinematic(feats))
    return out


def _determine_mood(cinematic_vector: np.ndarray) -> str:
    """
    Human-readable mood label from the 6D vector:
    [emotional_tone, pacing, texture, intensity, intimacy, darkness]
    Covers 8+ combinations.
    """
    v = np.asarray(cinematic_vector, dtype="float32").reshape(-1)
    if v.shape[0] != 6:
        raise ValueError("cinematic_vector must have 6 elements")

    emotional_tone, pacing, texture, intensity, intimacy, darkness = [float(x) for x in v.tolist()]

    hi = lambda x: x >= 0.7
    mid = lambda x: 0.4 <= x < 0.7
    lo = lambda x: x < 0.4

    if lo(pacing) and hi(darkness) and hi(intimacy):
        return "Quiet & Brooding"
    if hi(pacing) and hi(intensity):
        return "Tense & Visceral"
    if hi(emotional_tone) and lo(darkness):
        return "Warm & Hopeful"
    if hi(darkness) and hi(intensity) and mid(pacing):
        return "Dark & Intense"
    if lo(intensity) and hi(intimacy) and mid(emotional_tone):
        return "Tender & Intimate"
    if hi(texture) and lo(intensity) and lo(pacing):
        return "Sparse & Meditative"
    if lo(intimacy) and hi(pacing) and mid(intensity):
        return "Grand & Adventurous"
    if mid(emotional_tone) and mid(pacing) and mid(intensity):
        return "Balanced & Reflective"
    if hi(emotional_tone) and hi(pacing) and lo(darkness):
        return "Bright & Energetic"
    if lo(emotional_tone) and lo(pacing) and hi(darkness):
        return "Somber & Slow-burn"

    # Fallback: pick a coarse quadrant
    if darkness > emotional_tone and intensity >= pacing:
        return "Moody & Charged"
    if emotional_tone >= darkness and pacing >= intensity:
        return "Upbeat & Driving"
    return "Nuanced & Cinematic"

