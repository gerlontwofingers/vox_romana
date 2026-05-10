"""voice_preflight — check that required Piper voice models are installed.

Called automatically before every render when the engine includes Piper
(``both`` or ``piper``).  Collects the languages the show needs, finds
which Piper voices are already installed, and prints a clear warning
(with download instructions) for any language that has no installed voice.

Non-blocking: a missing voice is a warning, not a hard error.  The renderer
will fall back gracefully — but the warning gives users a chance to grab
the right model before a long render kicks off.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# URL pattern for the Piper voices HuggingFace repo.
_HF_PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/tree/main"

# Well-known Piper voice suggestions per language code.
# These are just hints — users can choose any available model.
_SUGGESTED_VOICES: dict[str, str] = {
    "es": "es_ES-davefx-medium",
    "fr": "fr_FR-siwis-low",
    "de": "de_DE-thorsten-medium",
    "pt": "pt_PT-tugão-medium",
    "it": "it_IT-riccardo-x_low",
    "nl": "nl_NL-mls-medium",
    "pl": "pl_PL-mls_6892-low",
    "ru": "ru_RU-irina-medium",
    "zh": "zh_CN-huayan-medium",
    "uk": "uk_UA-lada-x_low",
    "cs": "cs_CZ-jirka-medium",
    "fi": "fi_FI-harri-medium",
    "nb": "no_NO-talesyntese-medium",
    "ca": "ca_ES-upc_ona-medium",
    "hu": "hu_HU-anna-medium",
    "ro": "ro_RO-mihai-medium",
    "sk": "sk_SK-lili-medium",
    "da": "da_DK-talesyntese-medium",
}


def _lang_from_voice_id(voice_id: str) -> str | None:
    """Extract BCP-47 language prefix from a Piper voice id (e.g. 'es_ES-davefx-medium' → 'es')."""
    if not voice_id or "_" not in voice_id:
        return None
    head = voice_id.split("_", 1)[0]
    if not (2 <= len(head) <= 3 and head.isalpha()):
        return None
    return head.lower()


def _installed_voices(model_dir: Path) -> list[str]:
    """Return voice ids of all .onnx files found under model_dir."""
    if not model_dir.exists():
        return []
    return sorted(
        p.stem for p in model_dir.rglob("*.onnx")
        if not p.name.endswith(".onnx.json")
    )


def _languages_needed(show_data: dict) -> set[str]:
    """Collect the set of non-English language codes the show requires.

    Sources (in priority order):
      1. ``character_langs`` dict — per-character overrides.
      2. ``language`` field — show-level default.

    English is excluded because Piper ships a wide range of English models
    and the user almost certainly has at least one installed.
    """
    needed: set[str] = set()

    show_lang = (show_data.get("language") or "en").strip().lower()
    if show_lang and show_lang != "en":
        needed.add(show_lang)

    for lang in (show_data.get("character_langs") or {}).values():
        lang = (lang or "").strip().lower()
        if lang and lang != "en":
            needed.add(lang)

    return needed


def check_piper_voices(
    show_json: str | Path | dict,
    *,
    model_dir: str | Path = "piper-voices",
    engine: str = "both",
    quiet: bool = False,
) -> list[str]:
    """Check whether required Piper voices are installed.

    Args:
        show_json:  Path to the show JSON file, or an already-loaded dict.
        model_dir:  Directory to scan for .onnx voice model files.
        engine:     TTS engine in use.  If ``"kokoro"`` or ``"vc"``, Piper
                    is not involved and the check is skipped entirely.
        quiet:      If True, suppress output (logging still works).

    Returns:
        List of language codes that are missing Piper voices.
        Empty list when everything is covered (or the check was skipped).
    """
    # Only applies when Piper is part of the engine stack.
    if engine not in ("piper", "both"):
        return []

    # Load show data.
    if isinstance(show_json, dict):
        show_data = show_json
    else:
        path = Path(show_json)
        if not path.exists():
            logger.debug("voice_preflight: show JSON not found at %s — skipping", path)
            return []
        try:
            with path.open(encoding="utf-8") as fh:
                show_data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("voice_preflight: could not read show JSON (%s) — skipping", exc)
            return []

    needed = _languages_needed(show_data)
    if not needed:
        return []

    model_dir_path = Path(model_dir)
    installed = _installed_voices(model_dir_path)
    installed_langs: set[str] = {
        lang for v in installed if (lang := _lang_from_voice_id(v))
    }

    missing: list[str] = sorted(lang for lang in needed if lang not in installed_langs)
    if not missing:
        return []

    # Build a human-readable warning.
    try:
        from ..core.languages import LANGUAGE_NAMES
    except ImportError:
        LANGUAGE_NAMES = {}

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════╗",
        "║  VOICE PREFLIGHT WARNING — missing Piper voice models    ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        "  The show requires voices for the following language(s),",
        "  but no matching Piper .onnx model was found:",
        "",
    ]
    for lang in missing:
        lang_name = LANGUAGE_NAMES.get(lang, lang)
        suggestion = _SUGGESTED_VOICES.get(lang)
        lines.append(f"  • {lang_name} ({lang})")
        if suggestion:
            # Piper voices follow: <lang>/<lang_REGION>/<name>/<quality>/
            parts = suggestion.split("-")  # e.g. ["es_ES", "davefx", "medium"]
            if len(parts) >= 2:
                lang_region = parts[0]          # es_ES
                lang_prefix = lang_region.split("_")[0]  # es
                quality = parts[-1]             # medium
                name = "-".join(parts[1:-1])    # davefx
                hf_url = f"{_HF_PIPER_BASE}/{lang_prefix}/{lang_region}/{name}/{quality}"
            else:
                hf_url = _HF_PIPER_BASE
            lines.append(f"    Suggested: {suggestion}")
            lines.append(f"    Download:  {hf_url}")
        lines.append("")

    lines += [
        f"  Voices directory: {model_dir_path.resolve()}",
        "",
        "  The render will continue using fallback English voices for",
        "  these characters — but results will sound wrong.  Download",
        "  the .onnx + .onnx.json pair and place them anywhere under",
        "  the voices directory, then re-run.",
        "",
    ]

    warning_text = "\n".join(lines)
    if not quiet:
        print(warning_text)
    logger.warning("voice_preflight: missing Piper voices for languages: %s", missing)

    return missing
