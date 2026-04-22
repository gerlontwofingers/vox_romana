"""Language registry for vox_romana.

Defines the supported language codes, human-readable names, and utilities
for parsing the --out-lang CLI argument.

The supported_languages() function is a stub in Shot 1 — it always returns
["en"] because language-aware TTS engines are wired in Shot 3.  Once Shot 3
lands, supported_languages() will scan the installed Piper model directory
and return all language codes that have at least one voice model on disk.
"""

from __future__ import annotations

# ── Language metadata ────────────────────────────────────────────────────────

# BCP-47 language codes → human-readable display name.
# This is the master registry of languages vox_romana can (or will) support.
# A language being listed here does NOT guarantee TTS coverage — that depends
# on which Piper ONNX models are installed.  Kokoro only supports "en".
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ar": "Arabic",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "sv": "Swedish",
    "tr": "Turkish",
    "hi": "Hindi",
    "uk": "Ukrainian",
    "cs": "Czech",
    "fi": "Finnish",
    "nb": "Norwegian Bokmål",
    "ca": "Catalan",
    "hu": "Hungarian",
    "ro": "Romanian",
    "sk": "Slovak",
    "da": "Danish",
    "el": "Greek",
    "bg": "Bulgarian",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
}

# Convenience: the set of all valid language codes.
LANGUAGE_CODES: frozenset[str] = frozenset(LANGUAGE_NAMES)


# ── Availability ─────────────────────────────────────────────────────────────


def supported_languages() -> list[str]:
    """Return the list of language codes that have TTS voice coverage.

    Scans the configured Piper voices directory for installed ONNX models
    and parses each filename's ``<lang>_<COUNTRY>-<name>-<quality>`` prefix
    to determine which languages are available.  English (``"en"``) is
    always included because Kokoro provides English voices that don't
    require any model files on disk.

    Only languages also listed in ``LANGUAGE_NAMES`` are returned, so an
    accidental rogue ONNX file with an unknown prefix won't pollute the
    output.

    Returns:
        Sorted list of BCP-47 language codes with voice coverage.
        Example: ``["en", "es", "fr"]``.
    """
    langs: set[str] = {"en"}  # Kokoro always provides English.

    # Lazy imports to avoid a hard dependency on either module at import
    # time of the language registry (and to allow tests to monkey-patch).
    try:
        from vox_romana.core.paths import PIPER_VOICES_DIR
        from vox_romana.tts.engines.piper_engine import _lang_from_voice_id
    except Exception:
        return sorted(langs)

    if not PIPER_VOICES_DIR.exists():
        return sorted(langs)

    for onnx_path in PIPER_VOICES_DIR.rglob("*.onnx"):
        if onnx_path.name.endswith(".onnx.json"):
            continue
        code = _lang_from_voice_id(onnx_path.stem)
        if code and code in LANGUAGE_CODES:
            langs.add(code)

    return sorted(langs)


# ── CLI argument parsing ─────────────────────────────────────────────────────


def normalize_lang_arg(raw: str) -> list[str]:
    """Parse a raw --out-lang / --root-lang argument into a list of codes.

    Handles all the forms the user might type:

        "en"          → ["en"]
        "en,es"       → ["en", "es"]
        "en, es, fr"  → ["en", "es", "fr"]
        "all"         → supported_languages()

    Args:
        raw: The raw string value from argparse.

    Returns:
        Sorted, deduplicated list of valid BCP-47 language codes.

    Raises:
        ValueError: If any token is not a known language code (and not
                    "all").  The message lists all unknown tokens so the
                    user can fix everything in one go.
    """
    stripped = raw.strip()

    if stripped.lower() == "all":
        return supported_languages()

    tokens = [t.strip().lower() for t in stripped.split(",") if t.strip()]
    unknown = [t for t in tokens if t not in LANGUAGE_CODES]
    if unknown:
        valid_sample = ", ".join(sorted(LANGUAGE_CODES)[:10])
        raise ValueError(
            f"Unknown language code(s): {', '.join(unknown)}.  "
            f"Valid codes include: {valid_sample}, …  "
            f"Use --out-lang all to request every language with TTS coverage."
        )

    # Deduplicate while preserving first-seen order, then sort for stability.
    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return sorted(unique)


def language_display(codes: list[str]) -> str:
    """Format a list of language codes for display in the CLI banner.

    Examples:
        ["en"]           → "English (en)"
        ["en", "es"]     → "English (en), Spanish (es)"
        ["en", "es", "fr"] → "English (en), Spanish (es), French (fr)"

    Args:
        codes: List of BCP-47 language codes.

    Returns:
        Human-readable string.
    """
    parts = [f"{LANGUAGE_NAMES.get(c, c)} ({c})" for c in codes]
    return ", ".join(parts)
