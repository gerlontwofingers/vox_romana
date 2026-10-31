"""Studio instantiation prompts — one-sentence brief → full studio spec.

Shot 5.2-a of the Studio Instantiation feature.  A "studio" is the
showrunner-level identity (title, format, tone, writers' bible, target
audience) that a :class:`~vox_romana.core.catalog.SeriesRecord` already
carries.  Today that record is assembled by hand via ``--mode series
create``; this module supplies the STAGED inference that builds it from a
single sentence.

The inference runs in three deliberate stages (not one giant call) so each
layer can be constrained and validated independently:

    Stage 1  META     → title, show_type, tone, series_id slug, premise
    Stage 2  BIBLE    → the writers' bible (tone guide, motifs, canon)
    Stage 3  AUDIENCE → an :class:`AudienceProfile` (vocab, prior knowledge)

All three stages are mock-friendly: the only protocol a client must expose
is a ``.chat(messages, **kwargs) -> str`` method, identical to the
:class:`~vox_romana.core.client.DeepSeekClient` shape used by
:class:`~vox_romana.planning.episode_planner.EpisodePlanner`.

Constraint sources:
    - ``show_type`` is validated against
      :data:`vox_romana.core.router.SHOW_TYPES`.
    - ``vocabulary_level`` / ``education_level`` are validated against the
      enums in :mod:`vox_romana.core.audience`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from ..core.audience import (
    EDUCATION_LEVELS,
    VOCABULARY_LEVELS,
    AudienceProfile,
)
from ..core.router import SHOW_TYPES


# ─────────────────────────────────────────────────────────────────────────────
# Public dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StudioMeta:
    """Stage-1 output — the identifying metadata for a studio.

    Attributes:
        title:      Human-readable studio/series title.
        show_type:  One of the known show types (validated on parse).
        tone:       Free-form tone description (may be None).
        series_id:  URL-safe slug used as the series key.
        premise:    One-line description of what the studio produces.
    """

    title: str
    show_type: str
    series_id: str
    premise: str
    tone: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# LLM client protocol  (minimal — keeps tests trivial)
# ─────────────────────────────────────────────────────────────────────────────


class _ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — META
# ─────────────────────────────────────────────────────────────────────────────

_SHOW_TYPE_LIST = ", ".join(SHOW_TYPES)

META_SYSTEM_PROMPT = """\
You are a Studio Architect for a media production system.  Given a single \
sentence describing a show or content operation, you infer the identifying \
metadata needed to register it as a studio (series).

Return a single JSON object (and nothing else) with exactly these fields:

{
  "title":     "Human-readable studio/series title (one line, <= 80 chars).",
  "show_type": "One of: """ + _SHOW_TYPE_LIST + """.",
  "tone":      "Free-form tone description, or null if unclear.",
  "series_id": "URL-safe slug (lowercase, hyphens only) derived from the title.",
  "premise":   "One sentence: what this studio produces, for whom."
}

Rules:
- show_type MUST be exactly one of the allowed values.
- series_id MUST be lowercase letters, digits, and hyphens only; no spaces,
  no underscores, no other punctuation.
- Prefer the most specific show_type that fits the sentence.
- ALWAYS respond with valid JSON. Nothing else.
"""


def build_meta_user_prompt(brief: str) -> str:
    """Assemble the Stage-1 user message from the one-sentence brief."""
    return (
        f"Studio brief:\n"
        f"{brief.strip()}\n"
        f"\nInfer the studio metadata now.  Return ONLY the JSON object "
        f"described in the system message."
    )


def _slugify(text: str) -> str:
    """Coerce arbitrary text into a URL-safe slug (lowercase, hyphens)."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug


def parse_meta(raw: str) -> StudioMeta:
    """Parse Stage-1 output into a :class:`StudioMeta`.

    ``show_type`` is validated against the known show types; an
    unrecognized value falls back to ``"podcast"`` (the safest multi-host
    default) rather than raising, so a slightly-off LLM response doesn't
    abort instantiation.  ``series_id`` is slugified as a final guard.
    """
    data = json.loads(_extract_json_blob(raw))

    show_type = str(data.get("show_type") or "podcast").strip().lower()
    if show_type not in SHOW_TYPES:
        show_type = "podcast"

    title = str(data.get("title") or "").strip() or "Untitled Studio"
    series_id = _slugify(str(data.get("series_id") or title)) or "studio"

    tone = data.get("tone")
    tone = str(tone).strip() if tone else None

    premise = str(data.get("premise") or "").strip()

    return StudioMeta(
        title=title,
        show_type=show_type,
        tone=tone,
        series_id=series_id,
        premise=premise,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — BIBLE
# ─────────────────────────────────────────────────────────────────────────────

BIBLE_SYSTEM_PROMPT = """\
You are a showrunner writing a writers' bible for a media studio.  Given \
the studio's metadata, you produce the recurring rules that keep every \
episode consistent: tone guide, stylistic rules, recurring motifs, \
character canon (for fiction) or editorial voice (for non-fiction), and \
off-limits topics.

Return a single JSON object (and nothing else) with exactly this field:

{
  "bible": "The full writers' bible as multi-paragraph plain text."
}

Rules:
- The bible should be concrete and reusable — rules another writer could
  follow without seeing this prompt.
- Do NOT invent specific episode plots; describe the *standing* rules,
  voice, and canon instead.
- Keep it 150-350 words.
- ALWAYS respond with valid JSON. Nothing else.
"""


def build_bible_user_prompt(meta: StudioMeta) -> str:
    """Assemble the Stage-2 user message from Stage-1 metadata."""
    tone_line = f"Tone:       {meta.tone}" if meta.tone else "Tone:       (no fixed tone)"
    return (
        f"Studio title: {meta.title}\n"
        f"Show type:    {meta.show_type}\n"
        f"{tone_line}\n"
        f"Premise:      {meta.premise or '(none)'}\n"
        f"\nWrite the writers' bible now.  Return ONLY the JSON object "
        f"described in the system message."
    )


def parse_bible(raw: str) -> str:
    """Parse Stage-2 output into the bible text string."""
    data = json.loads(_extract_json_blob(raw))
    bible = str(data.get("bible") or "").strip()
    return bible


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — AUDIENCE
# ─────────────────────────────────────────────────────────────────────────────

_EDUCATION_LIST = ", ".join(EDUCATION_LEVELS)
_VOCAB_LIST = ", ".join(VOCABULARY_LEVELS)

AUDIENCE_SYSTEM_PROMPT = """\
You are an audience analyst for a media studio.  Given the studio's \
metadata, you infer the target audience and encode it as a structured \
profile that controls vocabulary, assumed prior knowledge, and tone.

Return a single JSON object (and nothing else) with exactly these fields:

{
  "age_range":        "Free-form age band, e.g. '18-35' or '40+' (or null).",
  "education_level":  "One of: """ + _EDUCATION_LIST + """ (or null).",
  "vocabulary_level": "One of: """ + _VOCAB_LIST + """.",
  "preferred_tone":   "Free-form tone preference (or null).",
  "prior_knowledge":  "What the listener is assumed to already know."
}

Rules:
- vocabulary_level MUST be one of the allowed values; default to
  'conversational' when unsure.
- education_level, when present, MUST be one of the allowed values.
- ALWAYS respond with valid JSON. Nothing else.
"""


def build_audience_user_prompt(meta: StudioMeta) -> str:
    """Assemble the Stage-3 user message from Stage-1 metadata."""
    tone_line = f"Tone:       {meta.tone}" if meta.tone else "Tone:       (no fixed tone)"
    return (
        f"Studio title: {meta.title}\n"
        f"Show type:    {meta.show_type}\n"
        f"{tone_line}\n"
        f"Premise:      {meta.premise or '(none)'}\n"
        f"\nInfer the audience profile now.  Return ONLY the JSON object "
        f"described in the system message."
    )


def parse_audience(raw: str) -> AudienceProfile:
    """Parse Stage-3 output into an :class:`AudienceProfile`.

    ``vocabulary_level`` and ``education_level`` are validated against the
    allowed enums; invalid values fall back to their defaults.
    """
    data = json.loads(_extract_json_blob(raw))

    vocab = str(data.get("vocabulary_level") or "conversational").strip().lower()
    if vocab not in VOCABULARY_LEVELS:
        vocab = "conversational"

    education = data.get("education_level")
    education = str(education).strip().lower() if education else None
    if education not in EDUCATION_LEVELS:
        education = None

    age_range = data.get("age_range")
    age_range = str(age_range).strip() if age_range else None

    preferred_tone = data.get("preferred_tone")
    preferred_tone = str(preferred_tone).strip() if preferred_tone else None

    prior_knowledge = data.get("prior_knowledge")
    prior_knowledge = str(prior_knowledge).strip() if prior_knowledge else None

    return AudienceProfile(
        age_range=age_range,
        education_level=education,
        vocabulary_level=vocab,
        preferred_tone=preferred_tone,
        prior_knowledge=prior_knowledge,
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction (shared by the three parsers)
# ─────────────────────────────────────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_blob(raw: str) -> str:
    """Pull the JSON object out of an LLM response.

    Tries (in order):
      1. A fenced ```json … ``` block.
      2. The substring from the first ``{`` to the last ``}``.

    Raises ValueError if neither yields JSON-shaped text.
    """
    m = _JSON_FENCE_RE.search(raw)
    if m:
        return m.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"StudioPrompts: no JSON object found in LLM response: {raw[:200]!r}"
        )
    return raw[start : end + 1]
