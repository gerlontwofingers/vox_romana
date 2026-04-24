"""The Voice Catalog — structured metadata for every available TTS voice.

If the engine is a recording booth and the renderer is the director,
this is the CASTING BOOK. It tells the director what each voice sounds
like so they can match characters to voices intelligently instead of
just going alphabetically down the roster.

THE CASTING DIRECTOR ANALOGY:

    Imagine casting a film. You don't just grab the first four actors
    who walk through the door. You look at the character descriptions:

        "Frank — gruff 50-year-old ex-detective"
        "Sophie — warm, quick-witted 30-something academic"

    Then you look at your roster of available actors and find the best
    fit based on age, tone, accent, and vocal quality. That's exactly
    what this module enables.

    Without the catalog, the renderer assigns voices round-robin by
    gender — which is like casting a movie by height. Functional, but
    you get Frank sounding like a teenager.

VOICE DESCRIPTOR FIELDS:

    Each voice entry contains:
        gender:    "male" or "female" — for gender-matching.
        accent:    "american" or "british" — for accent contrast.
        age:       "young", "middle", "mature" — rough age bracket.
        tone:      Brief tonal character (e.g., "warm", "authoritative").
        best_for:  When this voice shines (e.g., "narration", "debate").
        tags:      Flat list of searchable keywords for matching.

    The tags list is the primary matching surface. When the character
    profile says "gruff, older, authoritative", we score each voice
    by how many of its tags overlap with the character description.

ADDING NEW VOICES:

    When Kokoro adds new voices, add an entry to KOKORO_VOICE_CATALOG.
    Follow the existing pattern. The tags should be honest descriptors
    of what the voice ACTUALLY sounds like — not aspirational marketing.

USAGE:

    from vox_romana.tts.voice_catalog import (
        KOKORO_VOICE_CATALOG,
        match_voice,
        get_voices_by_gender,
    )

    # Find the best voice for a gruff older British man:
    voice_id = match_voice(
        "deep authoritative baritone, British accent, world-weary, 50s",
        gender="male",
        available_voices=["am_adam", "bm_george", "af_heart"],
    )
    # → "bm_george"

    # Get all female voices:
    females = get_voices_by_gender("female")
    # → {"af_heart": {...}, "af_alloy": {...}, ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
# VOICE DESCRIPTOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VoiceDescriptor:
    """Metadata describing what a TTS voice sounds like.

    This is the "headshot and demo reel" stapled to each voice's
    audition form. The matching algorithm reads these to decide
    which voice fits a character.

    Attributes:
        voice_id:  Engine-specific voice ID (e.g., "am_adam").
        gender:    "male" or "female".
        accent:    "american" or "british" (expandable to others).
        age:       "young" (20s), "middle" (30s-40s), "mature" (50s+).
        tone:      Brief tonal character — the elevator pitch for this
                   voice's personality. E.g., "warm and clear", "deep
                   and authoritative", "bright and energetic".
        best_for:  What this voice is best suited for — "narration",
                   "debate", "news anchor", "casual conversation", etc.
        tags:      Flat list of searchable keywords. These are the primary
                   matching surface when scoring a voice against a character
                   hint string. Keep them honest and descriptive.
        engine:    Which TTS engine owns this voice — "kokoro" or "piper".
                   Used by the MultiEngine wrapper to dispatch synthesis to
                   the right engine when both are loaded.
    """

    voice_id: str
    gender: str
    accent: str
    age: str
    tone: str
    best_for: str
    tags: tuple[str, ...] = ()
    engine: str = "kokoro"


# ═══════════════════════════════════════════════════════════════════════════
# KOKORO VOICE CATALOG
# ═══════════════════════════════════════════════════════════════════════════
# Naming convention: {accent_letter}{gender_letter}_{name}
#   a = American, b = British
#   f = female, m = male
#
# These descriptions are based on listening tests with the Kokoro model.
# They describe what each voice ACTUALLY sounds like, not what its name
# suggests. Your ears are the ground truth.

KOKORO_VOICE_CATALOG: dict[str, VoiceDescriptor] = {

    # ── American Female ──────────────────────────────────────────────

    "af_heart": VoiceDescriptor(
        voice_id="af_heart",
        gender="female",
        accent="american",
        age="middle",
        tone="warm, natural, clear",
        best_for="narration, general dialogue, warm characters",
        tags=("warm", "natural", "clear", "friendly", "conversational",
              "narrator", "approachable", "smooth"),
    ),
    "af_alloy": VoiceDescriptor(
        voice_id="af_alloy",
        gender="female",
        accent="american",
        age="middle",
        tone="neutral, professional, steady",
        best_for="news anchor, professional characters, exposition",
        tags=("neutral", "professional", "steady", "measured", "crisp",
              "anchor", "composed", "reliable"),
    ),
    "af_aoede": VoiceDescriptor(
        voice_id="af_aoede",
        gender="female",
        accent="american",
        age="young",
        tone="bright, expressive, melodic",
        best_for="animated discussion, young characters, enthusiasm",
        tags=("bright", "expressive", "melodic", "youthful", "energetic",
              "animated", "lively", "enthusiastic"),
    ),
    "af_jessica": VoiceDescriptor(
        voice_id="af_jessica",
        gender="female",
        accent="american",
        age="middle",
        tone="confident, articulate, polished",
        best_for="debate, academic characters, authority",
        tags=("confident", "articulate", "polished", "authoritative",
              "sharp", "academic", "precise", "intelligent"),
    ),
    "af_kore": VoiceDescriptor(
        voice_id="af_kore",
        gender="female",
        accent="american",
        age="young",
        tone="soft, thoughtful, reflective",
        best_for="introspective characters, contemplation, poetry",
        tags=("soft", "thoughtful", "reflective", "gentle", "contemplative",
              "quiet", "measured", "introspective"),
    ),
    "af_nicole": VoiceDescriptor(
        voice_id="af_nicole",
        gender="female",
        accent="american",
        age="middle",
        tone="crisp, direct, no-nonsense",
        best_for="reporters, factual delivery, direct characters",
        tags=("crisp", "direct", "no-nonsense", "matter-of-fact",
              "reporter", "factual", "blunt", "efficient"),
    ),
    "af_nova": VoiceDescriptor(
        voice_id="af_nova",
        gender="female",
        accent="american",
        age="young",
        tone="modern, versatile, engaging",
        best_for="podcasts, versatile roles, contemporary characters",
        tags=("modern", "versatile", "engaging", "contemporary",
              "podcast", "relatable", "dynamic", "adaptable"),
    ),
    "af_river": VoiceDescriptor(
        voice_id="af_river",
        gender="female",
        accent="american",
        age="middle",
        tone="calm, soothing, steady",
        best_for="narration, calming presence, mediator roles",
        tags=("calm", "soothing", "steady", "serene", "grounded",
              "mediator", "reassuring", "composed"),
    ),
    "af_sarah": VoiceDescriptor(
        voice_id="af_sarah",
        gender="female",
        accent="american",
        age="middle",
        tone="warm, conversational, relatable",
        best_for="casual dialogue, everyday characters, interviews",
        tags=("warm", "conversational", "relatable", "everyday",
              "casual", "approachable", "natural", "down-to-earth"),
    ),
    "af_sky": VoiceDescriptor(
        voice_id="af_sky",
        gender="female",
        accent="american",
        age="young",
        tone="airy, light, optimistic",
        best_for="cheerful characters, lighter segments, youth",
        tags=("airy", "light", "optimistic", "cheerful", "youthful",
              "upbeat", "breezy", "hopeful"),
    ),

    # ── American Male ────────────────────────────────────────────────

    "am_adam": VoiceDescriptor(
        voice_id="am_adam",
        gender="male",
        accent="american",
        age="middle",
        tone="deep, authoritative, commanding",
        best_for="debate lead, authority figures, serious characters",
        tags=("deep", "authoritative", "commanding", "serious",
              "gravitas", "leader", "anchor", "strong"),
    ),
    "am_echo": VoiceDescriptor(
        voice_id="am_echo",
        gender="male",
        accent="american",
        age="middle",
        tone="clear, neutral, adaptable",
        best_for="narration, versatile roles, news delivery",
        tags=("clear", "neutral", "adaptable", "versatile",
              "narrator", "professional", "balanced", "reliable"),
    ),
    "am_eric": VoiceDescriptor(
        voice_id="am_eric",
        gender="male",
        accent="american",
        age="young",
        tone="energetic, enthusiastic, dynamic",
        best_for="animated debate, passionate characters, sports",
        tags=("energetic", "enthusiastic", "dynamic", "passionate",
              "lively", "animated", "youthful", "spirited"),
    ),
    "am_fenrir": VoiceDescriptor(
        voice_id="am_fenrir",
        gender="male",
        accent="american",
        age="mature",
        tone="gruff, weathered, intense",
        best_for="tough characters, military, older authority figures",
        tags=("gruff", "weathered", "intense", "rough", "tough",
              "world-weary", "veteran", "gravelly", "mature"),
    ),
    "am_liam": VoiceDescriptor(
        voice_id="am_liam",
        gender="male",
        accent="american",
        age="young",
        tone="friendly, approachable, warm",
        best_for="casual conversation, interviews, relatable characters",
        tags=("friendly", "approachable", "warm", "casual",
              "relatable", "easygoing", "conversational", "young"),
    ),
    "am_michael": VoiceDescriptor(
        voice_id="am_michael",
        gender="male",
        accent="american",
        age="middle",
        tone="measured, thoughtful, professorial",
        best_for="academic debate, experts, analytical characters",
        tags=("measured", "thoughtful", "professorial", "analytical",
              "academic", "precise", "intellectual", "calm"),
    ),
    "am_onyx": VoiceDescriptor(
        voice_id="am_onyx",
        gender="male",
        accent="american",
        age="mature",
        tone="rich, resonant, dignified",
        best_for="narration, gravitas, distinguished characters",
        tags=("rich", "resonant", "dignified", "distinguished",
              "gravitas", "baritone", "elegant", "mature"),
    ),

    # ── British Female ───────────────────────────────────────────────

    "bf_emma": VoiceDescriptor(
        voice_id="bf_emma",
        gender="female",
        accent="british",
        age="middle",
        tone="clear, educated, composed",
        best_for="moderator, news anchor, authority with warmth",
        tags=("clear", "educated", "composed", "british", "polished",
              "moderator", "anchor", "authoritative", "poised"),
    ),
    "bf_isabella": VoiceDescriptor(
        voice_id="bf_isabella",
        gender="female",
        accent="british",
        age="middle",
        tone="refined, elegant, articulate",
        best_for="formal roles, narration, sophistication",
        tags=("refined", "elegant", "articulate", "british", "formal",
              "sophisticated", "graceful", "cultured"),
    ),
    "bf_lily": VoiceDescriptor(
        voice_id="bf_lily",
        gender="female",
        accent="british",
        age="young",
        tone="light, warm, personable",
        best_for="interviews, lighter segments, accessible delivery",
        tags=("light", "warm", "personable", "british", "accessible",
              "friendly", "youthful", "inviting"),
    ),

    # ── British Male ─────────────────────────────────────────────────

    "bm_daniel": VoiceDescriptor(
        voice_id="bm_daniel",
        gender="male",
        accent="british",
        age="middle",
        tone="crisp, professional, BBC-polished",
        best_for="news anchor, moderator, professional roles",
        tags=("crisp", "professional", "bbc", "polished", "british",
              "anchor", "moderator", "formal", "precise"),
    ),
    "bm_fable": VoiceDescriptor(
        voice_id="bm_fable",
        gender="male",
        accent="british",
        age="middle",
        tone="narrative, storytelling, evocative",
        best_for="narration, storytelling, documentary",
        tags=("narrative", "storytelling", "evocative", "british",
              "documentary", "literary", "resonant", "expressive"),
    ),
    "bm_george": VoiceDescriptor(
        voice_id="bm_george",
        gender="male",
        accent="british",
        age="mature",
        tone="deep, warm, distinguished gravitas",
        best_for="authority, senior characters, serious narration",
        tags=("deep", "warm", "distinguished", "gravitas", "british",
              "senior", "authoritative", "mature", "commanding"),
    ),
    "bm_lewis": VoiceDescriptor(
        voice_id="bm_lewis",
        gender="male",
        accent="british",
        age="young",
        tone="bright, conversational, engaging",
        best_for="casual dialogue, interviews, younger characters",
        tags=("bright", "conversational", "engaging", "british",
              "youthful", "casual", "lively", "approachable"),
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# PIPER VOICE CATALOG
# ═══════════════════════════════════════════════════════════════════════════
# Voice IDs match the .onnx stem in the piper-voices/ git clone:
#   piper-voices/en/en_US/lessac/medium/en_US-lessac-medium.onnx
#   → voice_id: "en_US-lessac-medium"
#
# Quality preference: high > medium > low (only best quality per speaker).
# Multi-speaker models (arctic, libritts, vctk, etc.) are excluded here
# because they require a numeric speaker_id to be useful.
#
# Gender/age annotations are based on voice name provenance and listening.

PIPER_VOICE_CATALOG: dict[str, VoiceDescriptor] = {

    # ── American English Female ──────────────────────────────────────

    "en_US-lessac-high": VoiceDescriptor(
        voice_id="en_US-lessac-high",
        gender="female",
        accent="american",
        age="middle",
        tone="clear, neutral, professional",
        best_for="narration, news delivery, professional characters",
        tags=("clear", "neutral", "professional", "narration",
              "smooth", "crisp", "natural", "anchor"),
    ),
    "en_US-amy-medium": VoiceDescriptor(
        voice_id="en_US-amy-medium",
        gender="female",
        accent="american",
        age="middle",
        tone="warm, conversational, approachable",
        best_for="interviews, dialogue, relatable characters",
        tags=("warm", "conversational", "approachable", "friendly",
              "natural", "everyday", "relatable", "casual"),
    ),
    "en_US-kathleen-low": VoiceDescriptor(
        voice_id="en_US-kathleen-low",
        gender="female",
        accent="american",
        age="mature",
        tone="measured, steady, composed",
        best_for="authority figures, formal narration, older characters",
        tags=("measured", "steady", "composed", "authoritative",
              "formal", "mature", "deliberate", "grounded"),
    ),
    "en_US-kristin-medium": VoiceDescriptor(
        voice_id="en_US-kristin-medium",
        gender="female",
        accent="american",
        age="middle",
        tone="bright, articulate, energetic",
        best_for="debate, discussion, animated dialogue",
        tags=("bright", "articulate", "energetic", "animated",
              "expressive", "lively", "confident", "debate"),
    ),
    "en_US-hfc_female-medium": VoiceDescriptor(
        voice_id="en_US-hfc_female-medium",
        gender="female",
        accent="american",
        age="young",
        tone="light, youthful, clear",
        best_for="younger characters, casual speech, modern content",
        tags=("light", "youthful", "clear", "young", "modern",
              "casual", "fresh", "accessible"),
    ),
    "en_US-ljspeech-high": VoiceDescriptor(
        voice_id="en_US-ljspeech-high",
        gender="female",
        accent="american",
        age="middle",
        tone="crisp, precise, broadcast-quality",
        best_for="audiobook narration, clear exposition, formal delivery",
        tags=("crisp", "precise", "broadcast", "audiobook",
              "narrator", "formal", "polished", "clear"),
    ),

    # ── American English Male ────────────────────────────────────────

    "en_US-ryan-high": VoiceDescriptor(
        voice_id="en_US-ryan-high",
        gender="male",
        accent="american",
        age="middle",
        tone="warm, authoritative, confident",
        best_for="debate lead, narration, authority figures",
        tags=("warm", "authoritative", "confident", "strong",
              "leader", "narration", "debate", "commanding"),
    ),
    "en_US-joe-medium": VoiceDescriptor(
        voice_id="en_US-joe-medium",
        gender="male",
        accent="american",
        age="middle",
        tone="casual, conversational, friendly",
        best_for="casual dialogue, interviews, everyday characters",
        tags=("casual", "conversational", "friendly", "approachable",
              "natural", "everyday", "relatable", "easygoing"),
    ),
    "en_US-danny-low": VoiceDescriptor(
        voice_id="en_US-danny-low",
        gender="male",
        accent="american",
        age="young",
        tone="energetic, youthful, direct",
        best_for="passionate arguments, younger characters, dynamic roles",
        tags=("energetic", "youthful", "direct", "passionate",
              "dynamic", "animated", "spirited", "young"),
    ),
    "en_US-norman-medium": VoiceDescriptor(
        voice_id="en_US-norman-medium",
        gender="male",
        accent="american",
        age="mature",
        tone="deep, deliberate, gravitas",
        best_for="serious authority, elder characters, gravitas",
        tags=("deep", "deliberate", "gravitas", "serious",
              "elder", "authoritative", "mature", "weighty"),
    ),
    "en_US-bryce-medium": VoiceDescriptor(
        voice_id="en_US-bryce-medium",
        gender="male",
        accent="american",
        age="middle",
        tone="smooth, professional, clear",
        best_for="news anchor, exposition, versatile roles",
        tags=("smooth", "professional", "clear", "versatile",
              "anchor", "polished", "measured", "reliable"),
    ),
    "en_US-john-medium": VoiceDescriptor(
        voice_id="en_US-john-medium",
        gender="male",
        accent="american",
        age="middle",
        tone="steady, measured, balanced",
        best_for="narration, analytical roles, moderator",
        tags=("steady", "measured", "balanced", "analytical",
              "narrator", "moderator", "calm", "intellectual"),
    ),
    "en_US-hfc_male-medium": VoiceDescriptor(
        voice_id="en_US-hfc_male-medium",
        gender="male",
        accent="american",
        age="young",
        tone="clear, crisp, modern",
        best_for="tech-forward content, younger characters, accessible delivery",
        tags=("clear", "crisp", "modern", "young", "contemporary",
              "fresh", "accessible", "tech"),
    ),
    "en_US-sam-medium": VoiceDescriptor(
        voice_id="en_US-sam-medium",
        gender="male",
        accent="american",
        age="middle",
        tone="robust, expressive, resonant",
        best_for="storytelling, character-driven dialogue, debate",
        tags=("robust", "expressive", "resonant", "storytelling",
              "character", "debate", "dynamic", "rich"),
    ),

    # ── British English Female ───────────────────────────────────────

    "en_GB-alba-medium": VoiceDescriptor(
        voice_id="en_GB-alba-medium",
        gender="female",
        accent="british",
        age="middle",
        tone="warm, articulate, natural",
        best_for="narration, dialogue, approachable British characters",
        tags=("warm", "articulate", "natural", "british", "approachable",
              "narrator", "friendly", "clear"),
    ),
    "en_GB-cori-high": VoiceDescriptor(
        voice_id="en_GB-cori-high",
        gender="female",
        accent="british",
        age="middle",
        tone="polished, refined, authoritative",
        best_for="BBC-style delivery, formal characters, professional roles",
        tags=("polished", "refined", "authoritative", "british", "bbc",
              "formal", "professional", "composed"),
    ),
    "en_GB-jenny_dioco-medium": VoiceDescriptor(
        voice_id="en_GB-jenny_dioco-medium",
        gender="female",
        accent="british",
        age="young",
        tone="bright, personable, engaging",
        best_for="interviews, lighter content, younger British characters",
        tags=("bright", "personable", "engaging", "british", "youthful",
              "accessible", "friendly", "lively"),
    ),
    "en_GB-southern_english_female-low": VoiceDescriptor(
        voice_id="en_GB-southern_english_female-low",
        gender="female",
        accent="british",
        age="mature",
        tone="measured, elegant, composed",
        best_for="formal narration, elder characters, distinguished roles",
        tags=("measured", "elegant", "composed", "british", "formal",
              "mature", "distinguished", "posh"),
    ),

    # ── British English Male ─────────────────────────────────────────

    "en_GB-alan-medium": VoiceDescriptor(
        voice_id="en_GB-alan-medium",
        gender="male",
        accent="british",
        age="middle",
        tone="warm, authoritative, clear",
        best_for="narration, authority figures, debate moderation",
        tags=("warm", "authoritative", "clear", "british", "narrator",
              "moderator", "confident", "professional"),
    ),
    "en_GB-northern_english_male-medium": VoiceDescriptor(
        voice_id="en_GB-northern_english_male-medium",
        gender="male",
        accent="british",
        age="middle",
        tone="gruff, direct, salt-of-the-earth",
        best_for="working-class characters, straight-talking roles, contrast",
        tags=("gruff", "direct", "northern", "british", "blunt",
              "earthy", "working-class", "salt-of-the-earth"),
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE TAGGING + UNIFIED CATALOG
# ═══════════════════════════════════════════════════════════════════════════
# VoiceDescriptor.engine defaults to "kokoro", which is correct for every
# entry in KOKORO_VOICE_CATALOG. We re-stamp PIPER_VOICE_CATALOG entries
# with engine="piper" via dataclasses.replace (descriptors are frozen).
# UNIFIED_VOICE_CATALOG merges both for the MultiEngine wrapper.

from dataclasses import replace as _replace

PIPER_VOICE_CATALOG = {
    vid: _replace(desc, engine="piper")
    for vid, desc in PIPER_VOICE_CATALOG.items()
}

# Sanity check at import time: no voice_id may appear in both catalogs.
# A collision would mean the MultiEngine couldn't unambiguously route a
# synthesis call. If you hit this, rename one of the conflicting voices.
_collisions = set(KOKORO_VOICE_CATALOG) & set(PIPER_VOICE_CATALOG)
if _collisions:
    raise RuntimeError(
        "Voice ID collision between KOKORO and PIPER catalogs: "
        f"{sorted(_collisions)}. Each voice_id must be unique across "
        "engines so MultiEngine can route synthesis correctly."
    )

UNIFIED_VOICE_CATALOG: dict[str, VoiceDescriptor] = {
    **KOKORO_VOICE_CATALOG,
    **PIPER_VOICE_CATALOG,
}


# ═══════════════════════════════════════════════════════════════════════════
# CATALOG QUERY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_voices_by_gender(
    gender: str,
    catalog: dict[str, VoiceDescriptor] | None = None,
) -> dict[str, VoiceDescriptor]:
    """Filter the catalog to voices of a given gender.

    Args:
        gender:   "male" or "female".
        catalog:  Voice catalog dict. Defaults to KOKORO_VOICE_CATALOG.

    Returns:
        Filtered dict of voice_id → VoiceDescriptor.
    """
    catalog = catalog or KOKORO_VOICE_CATALOG
    return {
        vid: desc for vid, desc in catalog.items()
        if desc.gender == gender.lower()
    }


def get_catalog_voice(
    voice_id: str,
    catalog: dict[str, VoiceDescriptor] | None = None,
) -> VoiceDescriptor | None:
    """Look up a single voice by ID.

    Args:
        voice_id: Engine-specific voice ID (e.g., "am_adam").
        catalog:  Voice catalog dict. Defaults to KOKORO_VOICE_CATALOG.

    Returns:
        The VoiceDescriptor, or None if not in the catalog.
    """
    catalog = catalog or KOKORO_VOICE_CATALOG
    return catalog.get(voice_id)


def match_voice(
    hint: str,
    *,
    gender: str | None = None,
    available_voices: list[str] | None = None,
    exclude: set[str] | None = None,
    catalog: dict[str, VoiceDescriptor] | None = None,
    position: int = 0,
) -> str | None:
    """Find the best voice for a character description.

    Scores each voice in the catalog by counting how many of its tags
    appear in the hint string. Higher score = better fit. Ties among
    voices with the same non-zero score are broken by catalog insertion
    order. When ALL candidates score zero (no tag overlap), the
    ``position`` index is used to pick a different voice per speaker so
    that identical-hint casts still get variety.

    Args:
        hint:             Natural-language character voice description.
                          E.g., "deep authoritative baritone, British,
                          world-weary, 50s". The more keywords, the
                          better the match.
        gender:           Filter to "male" or "female" voices only.
                          None = consider all voices.
        available_voices: Only consider these voice IDs. If None,
                          considers all voices in the catalog.
        exclude:          Voice IDs already assigned — skip these to
                          ensure every character gets a unique voice.
        catalog:          Voice catalog dict. Defaults to KOKORO_VOICE_CATALOG.
        position:         0-based index of this speaker in the cast. Used
                          as a tie-breaker when hint produces no tag overlap —
                          speaker 0 gets the first eligible voice, speaker 1
                          gets the second, etc.

    Returns:
        The best-matching voice_id string, or None if no voices
        are eligible (empty catalog, everything excluded, etc.).

    Example:
        >>> match_voice(
        ...     "gruff, older, authoritative, British",
        ...     gender="male",
        ...     exclude={"am_adam"},
        ... )
        'bm_george'
    """
    catalog = catalog or KOKORO_VOICE_CATALOG
    exclude = exclude or set()
    hint_lower = hint.lower()

    # Build the candidate pool: filter by gender, availability, exclusions.
    candidates: list[VoiceDescriptor] = []
    for vid, desc in catalog.items():
        if gender and desc.gender != gender.lower():
            continue
        if available_voices is not None and vid not in available_voices:
            continue
        if vid in exclude:
            continue
        candidates.append(desc)

    if not candidates:
        return None

    # Score each candidate by tag overlap with the hint string.
    # We check if each tag appears as a substring in the lowered hint.
    # This handles phrases like "world-weary" matching the tag "world-weary",
    # and also "authoritative" matching as a whole word.
    def _score(desc: VoiceDescriptor) -> int:
        return sum(1 for tag in desc.tags if tag in hint_lower)

    scores = [_score(c) for c in candidates]
    best_score = max(scores)
    if best_score > 0:
        # At least one tag matched — return the highest-scoring candidate.
        best = max(candidates, key=_score)
    else:
        # No tag overlap — use position to distribute voices across speakers
        # so a cast with empty hints doesn't all get the same voice.
        best = candidates[position % len(candidates)]
    return best.voice_id


def match_voices_for_cast(
    cast: dict[str, dict],
    available_voices: list[str] | None = None,
    catalog: dict[str, VoiceDescriptor] | None = None,
) -> dict[str, str]:
    """Assign unique voices to an entire cast, best-fit first.

    Processes characters in order, assigning the best available voice
    to each one while tracking exclusions so no two characters share
    a voice (unless the cast is larger than the available pool, in
    which case voices wrap).

    Args:
        cast:             Ordered dict of speaker_name → info dict.
                          Each info dict should have:
                            "gender": "male" or "female" (optional)
                            "voice_hint": character description (optional)
                          Missing fields are treated as unconstrained.
        available_voices: Only consider these voice IDs. If None,
                          uses the full catalog.
        catalog:          Voice catalog dict. Defaults to KOKORO_VOICE_CATALOG.

    Returns:
        Dict mapping speaker_name → voice_id for every speaker in cast.

    Example:
        >>> match_voices_for_cast({
        ...     "Frank": {"gender": "male", "voice_hint": "gruff, older, detective"},
        ...     "Sophie": {"gender": "female", "voice_hint": "warm, academic, 30s"},
        ... })
        {'Frank': 'am_fenrir', 'Sophie': 'af_heart'}
    """
    catalog = catalog or KOKORO_VOICE_CATALOG
    assigned: dict[str, str] = {}
    used: set[str] = set()

    for position, (speaker, info) in enumerate(cast.items()):
        gender = info.get("gender")
        hint = info.get("voice_hint", "")

        voice_id = match_voice(
            hint,
            gender=gender,
            available_voices=available_voices,
            exclude=used,
            catalog=catalog,
            position=position,
        )

        if voice_id is None:
            # All voices of this gender are used — reset exclusions for
            # this gender and try again (wrapping).
            voice_id = match_voice(
                hint,
                gender=gender,
                available_voices=available_voices,
                exclude=set(),
                catalog=catalog,
                position=position,
            )

        if voice_id:
            assigned[speaker] = voice_id
            used.add(voice_id)
        else:
            # Absolute fallback — shouldn't happen with a populated catalog.
            assigned[speaker] = "default"

    return assigned
