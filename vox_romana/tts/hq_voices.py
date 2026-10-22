"""High-quality voice pool: which voices may be auto-assigned, and fetching missing ones.

Some voices sound noticeably synthetic. This module defines a quality floor
that is based on published data rather than taste, so it can be audited:

* Kokoro: each voice has a published "overall grade" (training data quality and
  quantity) in the model's VOICES.md. Voices graded C or better are HQ.
* Piper: each voice ships in a quality tier. ``low`` and ``x_low`` voices are
  16 kHz; ``medium`` and ``high`` are 22.05 kHz. Only those two are HQ.
  English Piper voices must also be in the curated catalog (single-speaker,
  known gender); that excludes the multi-speaker corpora.
* Cloned voices (``vc_*``) are user-supplied and always allowed.

The floor never empties a pool. If a language has no HQ voice at all, its
normal voices are kept, because a wrong-language fallback sounds worse than a
lower-tier native voice.

Missing HQ voices can be downloaded on demand from the official Piper voice
repository, with a visible progress bar (see :func:`ensure_hq_voices`).
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

from .voice_catalog import KOKORO_VOICE_CATALOG, PIPER_VOICE_CATALOG, UNIFIED_VOICE_CATALOG

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# QUALITY RULES
# ═══════════════════════════════════════════════════════════════════════════

# Overall grades from https://huggingface.co/hexgrad/Kokoro-82M (VOICES.md).
# They estimate the quality and quantity of each voice's training data.
KOKORO_VOICE_GRADES: dict[str, str] = {
    "af_alloy": "C", "af_aoede": "C+", "af_bella": "A-", "af_heart": "A",
    "af_jessica": "D", "af_kore": "C+", "af_nicole": "B-", "af_nova": "C",
    "af_river": "D", "af_sarah": "C+", "af_sky": "C-",
    "am_adam": "F+", "am_echo": "D", "am_eric": "D", "am_fenrir": "C+",
    "am_liam": "D", "am_michael": "C+", "am_onyx": "D", "am_puck": "C+",
    "am_santa": "D-",
    "bf_alice": "D", "bf_emma": "B-", "bf_isabella": "C", "bf_lily": "D",
    "bm_daniel": "D", "bm_fable": "C", "bm_george": "C", "bm_lewis": "D+",
}
_GRADE_RANK = {
    "A": 11, "A-": 10, "B+": 9, "B": 8, "B-": 7, "C+": 6, "C": 5,
    "C-": 4, "D+": 3, "D": 2, "D-": 1, "F+": 0,
}
KOKORO_MIN_GRADE = "C"

HQ_PIPER_TIERS = frozenset({"medium", "high"})

# Escape hatch for individual voices that pass the rules above but still
# sound poor to a listener. Add voice ids here.
HQ_EXTRA_EXCLUDE: frozenset[str] = frozenset()

_PIPER_ID_RE = re.compile(r"^([a-z]{2,3})_([A-Z]{2})-(.+)-(x_low|low|medium|high)$")


def piper_quality(voice_id: str) -> str | None:
    """Return a Piper voice id's quality tier (``low``, ``medium``, ...) or None."""
    m = _PIPER_ID_RE.match(voice_id)
    return m.group(4) if m else None


def piper_language(voice_id: str) -> str | None:
    """Return the language code of a Piper voice id (``en_US-...`` -> ``en``)."""
    m = _PIPER_ID_RE.match(voice_id)
    return m.group(1) if m else None


def is_hq_voice(voice_id: str) -> bool:
    """True when *voice_id* meets the high-quality floor."""
    if voice_id in HQ_EXTRA_EXCLUDE:
        return False
    if voice_id.startswith("vc_"):
        return True
    grade = KOKORO_VOICE_GRADES.get(voice_id)
    if grade is not None:
        return _GRADE_RANK[grade] >= _GRADE_RANK[KOKORO_MIN_GRADE]
    tier = piper_quality(voice_id)
    if tier is None or tier not in HQ_PIPER_TIERS:
        return False
    if voice_id.startswith("en_"):
        return voice_id in PIPER_VOICE_CATALOG
    return True


def _group_key(voice_id: str) -> str:
    """Voices compete for HQ status within a language; Kokoro is English.

    Cloned voices get their own group so owning one never causes the normal
    English voices to be dropped.
    """
    if voice_id.startswith("vc_"):
        return "vc"
    return piper_language(voice_id) or "en"


def restrict_to_hq(voices: Iterable[str]) -> list[str]:
    """Filter *voices* to the HQ pool without ever emptying a language.

    Order is preserved so voice assignment stays deterministic.
    """
    voices = list(voices)
    groups: dict[str, list[str]] = {}
    for v in voices:
        groups.setdefault(_group_key(v), []).append(v)

    keep: set[str] = set()
    for key, members in groups.items():
        hq = [v for v in members if is_hq_voice(v)]
        if hq:
            keep.update(hq)
        else:
            keep.update(members)
            logger.warning(
                "hqvoices: no high-quality voice available for %r; "
                "keeping its %d normal voice(s).", key, len(members),
            )
    return [v for v in voices if v in keep]


# ═══════════════════════════════════════════════════════════════════════════
# ON-DEMAND DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════

PIPER_VOICES_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

# One medium-or-better voice per language, used when a show needs a language
# that has no HQ voice installed. All exist in the official repo.
HQ_DOWNLOAD_SUGGESTIONS: dict[str, str] = {
    "ar": "ar_JO-kareem-medium",
    "ca": "ca_ES-upc_ona-medium",
    "cs": "cs_CZ-jirka-medium",
    "da": "da_DK-talesyntese-medium",
    "de": "de_DE-thorsten-high",
    "es": "es_ES-davefx-medium",
    "fi": "fi_FI-harri-medium",
    "fr": "fr_FR-siwis-medium",
    "hi": "hi_IN-pratham-medium",
    "hu": "hu_HU-anna-medium",
    "it": "it_IT-paola-medium",
    "nl": "nl_NL-mls-medium",
    "no": "no_NO-talesyntese-medium",
    "pl": "pl_PL-gosia-medium",
    "pt": "pt_PT-tugão-medium",
    "ro": "ro_RO-mihai-medium",
    "ru": "ru_RU-irina-medium",
    "sk": "sk_SK-lili-medium",
    "sv": "sv_SE-lisa-medium",
    "tr": "tr_TR-dfki-medium",
    "uk": "uk_UA-ukrainian_tts-medium",
    "vi": "vi_VN-vais1000-medium",
    "zh": "zh_CN-huayan-medium",
}

_MIN_MODEL_BYTES = 1_000_000  # real models are tens of MB; guards HTML/LFS stubs


def piper_voice_urls(voice_id: str) -> tuple[str, str]:
    """Return the (model, config) download URLs for a Piper voice id."""
    m = _PIPER_ID_RE.match(voice_id)
    if m is None:
        raise ValueError(f"Not a Piper voice id: {voice_id!r}")
    lang, region, name, tier = m.groups()
    rel = f"{lang}/{lang}_{region}/{name}/{tier}/{voice_id}"
    base = f"{PIPER_VOICES_BASE_URL}/{urllib.parse.quote(rel)}"
    return f"{base}.onnx", f"{base}.onnx.json"


def piper_voice_local_dir(voice_id: str, model_dir: Path) -> Path:
    """Where a downloaded voice lives; mirrors the piper-voices repo layout."""
    m = _PIPER_ID_RE.match(voice_id)
    if m is None:
        raise ValueError(f"Not a Piper voice id: {voice_id!r}")
    lang, region, name, tier = m.groups()
    return Path(model_dir) / lang / f"{lang}_{region}" / name / tier


def _human_mb(n: float) -> str:
    return f"{n / 1_048_576:.1f} MB"


class _ProgressBar:
    """Minimal stderr progress bar; degrades to periodic lines when piped."""

    def __init__(self, label: str, total: int | None, stream: TextIO) -> None:
        self._label = label
        self._total = total
        self._stream = stream
        self._tty = bool(getattr(stream, "isatty", lambda: False)())
        self._last_draw = 0.0
        self._last_pct_printed = -1

    def update(self, done: int) -> None:
        if not self._total:
            return
        pct = int(done * 100 / self._total)
        if self._tty:
            now = time.monotonic()
            if now - self._last_draw < 0.1 and done < self._total:
                return
            self._last_draw = now
            filled = pct * 24 // 100
            bar = "#" * filled + "." * (24 - filled)
            self._stream.write(
                f"\r    {self._label}  [{bar}] {pct:3d}%  "
                f"{_human_mb(done)} / {_human_mb(self._total)}"
            )
            self._stream.flush()
        elif pct >= self._last_pct_printed + 25:
            self._last_pct_printed = pct - pct % 25
            self._stream.write(
                f"    {self._label}: {self._last_pct_printed}% "
                f"({_human_mb(done)} / {_human_mb(self._total)})\n"
            )
            self._stream.flush()

    def finish(self) -> None:
        if self._tty:
            self._stream.write("\n")
            self._stream.flush()


def download_piper_voice(
    voice_id: str,
    model_dir: str | Path,
    *,
    show_progress: bool = True,
    stream: TextIO | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> bool:
    """Download a Piper voice (model + config) into *model_dir*.

    Files are written to ``.part`` temporaries and renamed only after the size
    matches the server's Content-Length, so an interrupted download never
    leaves a half-written model that the engine would try to load.

    Returns:
        True on success, False on any failure (a warning is logged).
    """
    stream = stream or sys.stderr
    target_dir = piper_voice_local_dir(voice_id, Path(model_dir))
    model_url, config_url = piper_voice_urls(voice_id)
    model_path = target_dir / f"{voice_id}.onnx"
    config_path = target_dir / f"{voice_id}.onnx.json"
    partials: list[Path] = []

    try:
        target_dir.mkdir(parents=True, exist_ok=True)

        # Config first: tiny, and proves the voice id exists before the big file.
        tmp_cfg = config_path.with_name(config_path.name + ".part")
        partials.append(tmp_cfg)
        with opener(config_url, timeout=60) as resp:
            tmp_cfg.write_bytes(resp.read())
        json.loads(tmp_cfg.read_text(encoding="utf-8"))

        tmp_model = model_path.with_name(model_path.name + ".part")
        partials.append(tmp_model)
        with opener(model_url, timeout=60) as resp:
            length = resp.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            bar = _ProgressBar(voice_id, total, stream) if show_progress else None
            written = 0
            with tmp_model.open("wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
                    written += len(chunk)
                    if bar:
                        bar.update(written)
            if bar:
                bar.finish()

        if total is not None and written != total:
            raise OSError(f"incomplete download ({written} of {total} bytes)")
        if written < _MIN_MODEL_BYTES:
            raise OSError(f"downloaded file is too small to be a voice model ({written} bytes)")

        tmp_cfg.replace(config_path)
        tmp_model.replace(model_path)
        return True
    except (OSError, ValueError, urllib.error.URLError) as exc:
        logger.warning("hqvoices: could not download %s: %s", voice_id, exc)
        return False
    finally:
        for p in partials:
            p.unlink(missing_ok=True)


def _installed_piper_voices(model_dir: Path) -> set[str]:
    if not model_dir.exists():
        return set()
    return {
        p.stem for p in model_dir.rglob("*.onnx")
        if not p.name.endswith(".onnx.json")
    }


def _primary_language(code: str | None) -> str:
    return (code or "en").strip().lower().replace("_", "-").split("-", 1)[0] or "en"


def plan_hq_downloads(
    show_data: dict,
    installed: set[str],
    *,
    engine: str,
) -> list[tuple[str, str]]:
    """Decide which missing HQ Piper voices this show needs.

    Two triggers:
      * English speakers of a gender outnumber the HQ voices of that gender
        that are available (Kokoro counts when the engine includes it).
      * The show uses a language with no HQ Piper voice installed.

    Returns:
        ``[(voice_id, reason), ...]`` in download order; empty if nothing is needed.
    """
    if engine not in ("piper", "both"):
        return []

    plan: list[tuple[str, str]] = []
    planned: set[str] = set()

    # ── English cast vs. HQ pool, per gender ──────────────────────────
    language = show_data.get("language")
    character_langs = show_data.get("character_langs") or {}
    need = {"male": 0, "female": 0}
    for name, gender in (show_data.get("voice_casting") or {}).items():
        if _primary_language(character_langs.get(name) or language) != "en":
            continue
        if gender in need:
            need[gender] += 1

    pool_ids = {v for v in installed if v in PIPER_VOICE_CATALOG and is_hq_voice(v)}
    if engine == "both":
        pool_ids |= {v for v in KOKORO_VOICE_CATALOG if is_hq_voice(v)}

    for gender, count in need.items():
        have = sum(1 for v in pool_ids if UNIFIED_VOICE_CATALOG[v].gender == gender)
        deficit = count - have
        if deficit <= 0:
            continue
        candidates = [
            v for v, d in PIPER_VOICE_CATALOG.items()
            if d.gender == gender and is_hq_voice(v)
            and v not in installed and v not in planned
        ]
        candidates.sort(key=lambda v: piper_quality(v) != "high")  # stable: high first
        for v in candidates[:deficit]:
            plan.append((v, f"the show has {count} {gender} English speaker(s) "
                            f"but only {have} high-quality {gender} voice(s) available"))
            planned.add(v)

    # ── Languages with no HQ voice installed ──────────────────────────
    langs = {_primary_language(language)} | {
        _primary_language(c) for c in character_langs.values()
    }
    for lang in sorted(langs - {"en"}):
        if any(piper_language(v) == lang and is_hq_voice(v) for v in installed):
            continue
        suggestion = HQ_DOWNLOAD_SUGGESTIONS.get(lang)
        if suggestion and suggestion not in installed and suggestion not in planned:
            plan.append((suggestion, f"no high-quality {lang!r} voice is installed"))
            planned.add(suggestion)

    return plan


def _load_show(show_json: str | Path | dict) -> dict | None:
    if isinstance(show_json, dict):
        return show_json
    try:
        with Path(show_json).open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("hqvoices: could not read show JSON (%s); skipping voice check", exc)
        return None


def ensure_hq_voices(
    show_json: str | Path | dict,
    *,
    engine: str,
    model_dir: str | Path,
    quiet: bool = False,
    downloader: Callable[..., bool] = download_piper_voice,
    stream: TextIO | None = None,
) -> list[str]:
    """Download any HQ Piper voices the show needs but the machine lacks.

    Always tells the user before downloading (even with ``quiet``) and shows a
    progress bar per file unless ``quiet``. Failures are non-fatal: the
    renderer falls back to the voices it has.

    Returns:
        Voice ids that were downloaded successfully.
    """
    show_data = _load_show(show_json)
    if show_data is None:
        return []
    model_dir = Path(model_dir)
    plan = plan_hq_downloads(show_data, _installed_piper_voices(model_dir), engine=engine)
    if not plan:
        return []

    stream = stream or sys.stderr
    print(
        f"\n  High-quality voices: {len(plan)} voice(s) are missing and will be "
        f"downloaded to\n  {model_dir} (use --no-hqvoices to skip):",
        file=stream,
    )
    for voice_id, reason in plan:
        print(f"    - {voice_id}: {reason}", file=stream)

    done: list[str] = []
    for voice_id, _ in plan:
        if downloader(voice_id, model_dir, show_progress=not quiet, stream=stream):
            done.append(voice_id)
    failed = len(plan) - len(done)
    print(
        f"  Downloaded {len(done)} of {len(plan)} voice(s)."
        + (f" {failed} failed; continuing with the voices available.\n" if failed else "\n"),
        file=stream,
    )
    return done
