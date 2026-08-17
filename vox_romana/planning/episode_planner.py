"""EpisodePlanner — auto-write the brief for the next episode in a series.

The series workflow before this module:

    1. ``--mode series create``  → series record + writers'-bible.
    2. ``--mode series approve``  → asset pool (voices / music / visuals).
    3. *(human writes a brief by hand)*  ← the friction point.
    4. ``run.py prompts/ep07.txt --series SERIES_ID``  → episode produced.

EpisodePlanner removes step 3.  Given a series ID it reads:

    - the writers'-bible (``series.bible``),
    - the recent episode summaries (``Catalog.get_recent_episode_summaries``),
    - the next episode number (``Catalog.next_episode_number``),
    - an optional ``--topic`` constraint from the user,

…then asks an LLM to draft a fresh brief that fits the series voice but
does not repeat any prior episode.  The result is an :class:`EpisodeBrief`
dataclass containing the title, angle, hooks, an estimated duration, and
the actual ``prompt_text`` that would be saved to disk and fed back into
``--mode generate``.

THIS MODULE IS LLM-DRIVEN BUT MOCK-FRIENDLY.  The single public entry
point is :meth:`EpisodePlanner.plan_next`, which takes a fully-loaded
``DeepSeekClient`` (or anything with the same ``.chat()`` shape).  Tests
inject a stub client so no API key is needed.

Usage from the CLI runner (see ``vox_romana/cli/series.py``)::

    planner = EpisodePlanner(client)
    brief = planner.plan_next(
        series=series_record,
        recent_episodes=cat.get_recent_episode_summaries(series_id, n=5),
        next_episode_number=cat.next_episode_number(series_id),
        topic=args.topic,
    )
    brief_path.write_text(brief.prompt_text, encoding="utf-8")
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EpisodeBrief:
    """The output of :meth:`EpisodePlanner.plan_next`.

    Holds both structured planning fields (so callers can display a nice
    summary) and the unstructured ``prompt_text`` that gets written to
    disk and fed back into ``--mode generate``.

    Attributes:
        title:               One-line working title for the episode.
        angle:               One-paragraph description of the spin or
                             argument the episode advances.
        hooks:               Short list of opening hooks / cold-open
                             ideas.  Empty list when none were proposed.
        guests:              Suggested guest characters (debate/podcast
                             only).  Empty list for solo formats.
        estimated_duration:  Target duration in minutes; derived from the
                             show type when the LLM doesn't propose one.
        prompt_text:         The full plain-text brief, ready to write
                             to a .txt file and pass to ``--mode generate``
                             as the positional prompt argument.
        episode_number:      The number this brief is intended to fill,
                             carried through for downstream filename
                             generation.
        topic:               Echo of the user's ``--topic`` constraint
                             (empty string when none was given).
    """

    title: str
    angle: str
    prompt_text: str
    episode_number: int
    hooks: list[str] = field(default_factory=list)
    guests: list[str] = field(default_factory=list)
    estimated_duration: float = 15.0
    topic: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# LLM client protocol  (minimal — keeps tests trivial)
# ─────────────────────────────────────────────────────────────────────────────


class _ChatClient(Protocol):
    """Anything with a DeepSeekClient-shaped ``.chat()`` method.

    Tests pass a plain mock; production code passes
    :class:`vox_romana.core.client.DeepSeekClient`.
    """

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """You are the Series Planner — a senior producer who decides \
what the next episode of an ongoing show should be about.

You are given:
  - The series writers'-bible (tone, world facts, recurring characters).
  - Summaries of the most recent episodes (so you do not repeat them).
  - The episode number this brief is for.
  - Optionally, a topic the showrunner wants explored next.

Your job is to draft ONE compelling brief for the next episode.  The
brief must:
  1. Stay faithful to the bible's tone and any canon facts.
  2. Avoid duplicating angles, framings, or core arguments from the
     recent-episode list.
  3. Propose a tight focus — one strong angle, not three weak ones.
  4. End with a plain-text "BRIEF:" section that reads naturally as a
     standalone prompt (no JSON, no markdown headings inside it).

Return your answer as a single JSON object (and nothing else) with these
fields:

    {
      "title":               "Working title (one line, ≤ 80 chars).",
      "angle":               "One-paragraph spin or argument.",
      "hooks":               ["Cold-open idea 1", "..."],
      "guests":              ["Suggested guest 1 (or empty list)"],
      "estimated_duration":  15,
      "brief":               "Plain-text brief — this becomes the prompt file."
    }

The "brief" field MUST be a self-contained brief.  Do not reference
"the previous episodes" or "as we discussed before" — the production
pipeline does not see your reasoning."""


def _format_recent_episodes(episodes: list[dict[str, Any]]) -> str:
    """Render the recent-episode list as a compact markdown bullet list.

    Used inside the user prompt so the LLM can scan past episodes
    quickly.  Returns ``"(none yet — this is episode 1)"`` when the
    list is empty.
    """
    if not episodes:
        return "(none yet — this is episode 1)"
    lines: list[str] = []
    for ep in episodes:
        n = ep.get("episode_number") or "?"
        title = ep.get("title") or "(untitled)"
        # Trim long briefs to keep the prompt compact.
        brief = (ep.get("brief") or "").strip().replace("\n", " ")
        if len(brief) > 200:
            brief = brief[:200].rstrip() + "…"
        lines.append(f"  - [{n}] {title}\n      brief: {brief or '(empty)'}")
    return "\n".join(lines)


def _build_user_prompt(
    *,
    series_title: str,
    series_show_type: str | None,
    series_tone: str | None,
    bible: str,
    recent_episodes: list[dict[str, Any]],
    next_episode_number: int,
    topic: str | None,
) -> str:
    """Assemble the user message sent to the LLM."""
    show_type_line = f"Show type:    {series_show_type}" if series_show_type else "Show type:    (any)"
    tone_line = f"Tone:         {series_tone}" if series_tone else "Tone:         (no fixed tone)"
    bible_block = bible.strip() or "(empty bible — fall back to general best practices)"
    topic_line = (
        f"Topic constraint:\n  {topic.strip()}\n"
        if topic and topic.strip()
        else "Topic constraint:\n  (none — pick the most compelling fresh angle)\n"
    )
    return (
        f"Series title: {series_title}\n"
        f"{show_type_line}\n"
        f"{tone_line}\n"
        f"Next episode: #{next_episode_number}\n"
        f"\n"
        f"Writers'-bible:\n"
        f"───────────────\n"
        f"{bible_block}\n"
        f"───────────────\n"
        f"\n"
        f"Recent episodes (most recent first):\n"
        f"{_format_recent_episodes(recent_episodes)}\n"
        f"\n"
        f"{topic_line}"
        f"\n"
        f"Draft the brief now.  Return ONLY the JSON object described "
        f"in the system message."
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction (the LLM occasionally wraps with markdown fences)
# ─────────────────────────────────────────────────────────────────────────────


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_blob(raw: str) -> str:
    """Pull the JSON object out of an LLM response.

    Tries (in order):
      1. A fenced ```json … ``` block.
      2. The substring from the first ``{`` to the last ``}``.

    Raises ValueError if neither yields parseable JSON-shaped text.
    """
    m = _JSON_FENCE_RE.search(raw)
    if m:
        return m.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"EpisodePlanner: no JSON object found in LLM response: {raw[:200]!r}"
        )
    return raw[start : end + 1]


# ─────────────────────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────────────────────


# Default duration per show type when the LLM doesn't propose one.
_DEFAULT_DURATIONS: dict[str, float] = {
    "debate": 20.0,
    "news": 12.0,
    "podcast": 30.0,
    "audiobook": 60.0,
    "lecture": 30.0,
}


class EpisodePlanner:
    """LLM-backed agent that drafts the brief for a series' next episode.

    Stateless — one client can plan many series.  The single public
    method is :meth:`plan_next`.

    Attributes:
        _client: The chat client used for LLM calls.  Tests inject a
                 mock; production passes a DeepSeekClient.
    """

    def __init__(self, client: _ChatClient) -> None:
        self._client = client

    def plan_next(
        self,
        *,
        series_title: str,
        series_show_type: str | None,
        series_tone: str | None,
        bible: str,
        recent_episodes: list[dict[str, Any]],
        next_episode_number: int,
        topic: str | None = None,
        temperature: float = 0.8,
    ) -> EpisodeBrief:
        """Generate the next-episode brief.

        Args:
            series_title:        From ``SeriesRecord.title``.
            series_show_type:    From ``SeriesRecord.show_type`` — used
                                 to pick a default duration when the LLM
                                 omits one.
            series_tone:         From ``SeriesRecord.tone`` — included in
                                 the prompt so the LLM matches voice.
            bible:               From ``SeriesRecord.bible``.
            recent_episodes:     Output of
                                 ``Catalog.get_recent_episode_summaries``.
            next_episode_number: Output of ``Catalog.next_episode_number``.
            topic:               Optional ``--topic`` constraint from the
                                 user.  ``None`` or empty → free pick.
            temperature:         LLM temperature (default 0.8 — we want
                                 some creative range here).

        Returns:
            An :class:`EpisodeBrief` ready to be written to disk and
            handed back to ``--mode generate``.
        """
        user_prompt = _build_user_prompt(
            series_title=series_title,
            series_show_type=series_show_type,
            series_tone=series_tone,
            bible=bible,
            recent_episodes=recent_episodes,
            next_episode_number=next_episode_number,
            topic=topic,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        logger.debug("EpisodePlanner: calling LLM for episode #%d", next_episode_number)
        raw = self._client.chat(messages, temperature=temperature)

        try:
            blob = _extract_json_blob(raw)
            data = json.loads(blob)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"EpisodePlanner: LLM did not return valid JSON: {exc}.  "
                f"Response head: {raw[:200]!r}"
            ) from exc

        title = str(data.get("title") or "").strip() or f"Episode {next_episode_number}"
        angle = str(data.get("angle") or "").strip()
        brief_text = str(data.get("brief") or "").strip()
        if not brief_text:
            raise RuntimeError(
                "EpisodePlanner: LLM response is missing the required 'brief' field."
            )

        hooks = [str(h).strip() for h in (data.get("hooks") or []) if str(h).strip()]
        guests = [str(g).strip() for g in (data.get("guests") or []) if str(g).strip()]

        # Estimated duration: prefer the LLM's number, fall back to the
        # show-type default, then to a generic 15 min.
        try:
            est = float(data.get("estimated_duration") or 0)
        except (TypeError, ValueError):
            est = 0.0
        if est <= 0:
            est = _DEFAULT_DURATIONS.get(series_show_type or "", 15.0)

        return EpisodeBrief(
            title=title,
            angle=angle,
            prompt_text=brief_text,
            episode_number=next_episode_number,
            hooks=hooks,
            guests=guests,
            estimated_duration=est,
            topic=(topic or "").strip(),
        )
