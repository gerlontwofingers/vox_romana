"""Show router — classifies a brief into a show type.

Given a plain-English brief, the router makes a single LLM call to decide
which show pipeline should handle it: debate, news, or (eventually) others.

This is the front door's bouncer. You hand it a brief, it tells you which
room to go to.

    >>> router = ShowRouter(client)
    >>> result = router.classify("Four economists debate UBI")
    >>> result.show_type
    'debate'

The classification is deliberately simple: one LLM call, one JSON response.
No chain, no follow-up. Speed matters here because the user is waiting for
the real pipeline to start.
"""

from __future__ import annotations

import json
import re

from .client import DeepSeekClient

# ── Known show types ─────────────────────────────────────────────────────
SHOW_TYPES = ("debate", "news", "audiobook", "podcast", "lecture", "radio_drama")

# ── System prompt ────────────────────────────────────────────────────────
ROUTER_SYSTEM_PROMPT = """\
You are a show type classifier for a media production system.

Given a user's brief (a plain-English description of the content they want
to produce), you decide TWO things:
  1. Which show type pipeline should handle it.
  2. Whether the brief needs background research before generation.

Available show types:
  - "debate": Two or more participants argue, discuss, or deliberate a topic.
    Includes Oxford-style debates, panel discussions, crossfire formats,
    roundtables, and any adversarial or multi-perspective dialogue.
  - "news": A news broadcast, newscast, or bulletin. Anchors present
    segments covering current events, topics, or stories. Includes
    breaking news, weekly roundups, and magazine-style shows.
  - "audiobook": A narrated audiobook, story, or documentary. A single
    narrator (or small cast) reads through structured content chapter by
    chapter. Includes fiction, non-fiction, self-help, history, science
    explainers, and any long-form narrated audio. Choose THIS for narrative
    or storytelling content; choose "lecture" for explicitly didactic
    teaching content.
  - "lecture": A single-voice educational/didactic talk where a professor
    or teacher walks listeners through a topic with structured pedagogy:
    introduce, define, illustrate, recap. Includes university lectures,
    masterclasses, tutorials, and explainers framed as teaching. Key
    difference from audiobook: lectures TEACH (with learning objectives,
    analogies, exercises) while audiobooks NARRATE.
  - "podcast": A multi-host conversational podcast. Two or more hosts discuss
    topics in a casual, personality-driven format. Includes talk shows,
    roundtable discussions, interview pods, comedy panels, and any
    multi-host audio where the tone is conversational rather than
    adversarial. Key difference from debate: podcasts are collaborative
    conversations, not structured arguments.
  - "radio_drama": A scripted multi-voice narrative fiction episode in the
    radio-drama tradition (old-time radio serials, audio drama, noir,
    sci-fi, horror, sitcom). Includes dramatized stories with a cast of
    characters, scenes, an announcer cold-open/sign-off, and music/SFX
    cues. Key difference from audiobook: radio dramas are MULTI-VOICE
    scripted fiction (full cast, dialogue-driven scenes), while audiobooks
    are single-narrator (or small-cast) narration of a longer work.

Research detection — set needs_research to true when the brief:
  - Asks for real statistics, data, facts, evidence, or citations.
  - Names specific real-world events, policies, studies, or programs.
  - Covers factual topics where accuracy matters (economics, science,
    history, policy, current events).
  - Explicitly requests research ("use real data", "cite sources").
  - References specific real shows, books, games, etc. where the
    participants need verifiable details (episode counts, plot points, etc.).
  Set it to false when the brief:
  - Is purely fictional or hypothetical ("aliens vs robots").
  - Is opinion/values-based with no factual grounding needed.
  - Is clearly a creative exercise where making things up is the point.

Respond with ONLY a JSON object:
{
  "show_type": "<debate|news|audiobook|podcast|lecture|radio_drama>",
  "confidence": <0.0-1.0>,
  "needs_research": <true|false>,
  "reasoning": "<one sentence explaining both decisions>"
}

Rules:
- If the brief mentions "debate", "argue", "discuss", "panel", "versus",
  "pro/con", or multiple named participants taking sides, choose "debate".
- If it mentions "news", "headlines", "breaking", "anchor", "broadcast",
  "bulletin", "report", "coverage", or "roundup", choose "news".
- If it mentions "audiobook", "story", "narrate", "novel", "chapter",
    "essay", "documentary", "read aloud", or describes a single-narrator
    long-form narrative, choose "audiobook".
- If it mentions "lecture", "masterclass", "course", "tutorial",
    "explainer", "teach", "professor delivering", "walk through", or
    describes structured pedagogical content with explicit learning goals,
    choose "lecture".
- If it mentions "podcast", "pod", "show", "hosts", "episode", "panel
  discussion" (casual), "talk show", "roundtable" (conversational),
  or describes multiple hosts having a casual conversation, choose
  "podcast".
- If it mentions "radio drama", "audio drama", "radio play", "old-time
  radio", "dramatized", "scripted fiction", "cast of characters", "scenes",
  "announcer", or describes a multi-voice fictional story performed as a
  play, choose "radio_drama".
- When ambiguous, lean toward "debate" if the brief emphasizes opposing
  viewpoints, "news" if it emphasizes information delivery, "audiobook"
  if it emphasizes narration or storytelling, "podcast" if it
  emphasizes casual multi-host conversation, or "radio_drama" if it
  emphasizes performed multi-voice fiction.
- When in doubt about research, lean toward true — it's better to have
  facts and not need them than to wing it without preparation.
- Always respond with valid JSON. Nothing else.
"""

# ── User prompt template ────────────────────────────────────────────────
ROUTER_USER_TEMPLATE = "Classify this brief:\n\n{brief}"


class RouterResult:
    """The result of classifying a brief."""

    __slots__ = ("show_type", "confidence", "needs_research", "reasoning")

    def __init__(
        self,
        show_type: str,
        confidence: float,
        reasoning: str,
        needs_research: bool = False,
    ):
        if show_type not in SHOW_TYPES:
            raise ValueError(
                f"Unknown show type: {show_type!r}. "
                f"Expected one of: {SHOW_TYPES}"
            )
        self.show_type = show_type
        self.confidence = max(0.0, min(1.0, confidence))
        self.needs_research = needs_research
        self.reasoning = reasoning

    def __repr__(self) -> str:
        return (
            f"RouterResult(show_type={self.show_type!r}, "
            f"confidence={self.confidence:.2f}, "
            f"needs_research={self.needs_research})"
        )


def parse_router_response(raw: str) -> RouterResult:
    """Parse the LLM's JSON response into a RouterResult.

    Handles markdown fences, trailing commas, and other LLM quirks.
    """
    # Strip markdown code fences
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try stripping trailing commas (common LLM mistake)
        cleaned = re.sub(r",\s*}", "}", text)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        data = json.loads(cleaned)  # Let this raise if still bad

    show_type = data.get("show_type", "").strip().lower()
    confidence = float(data.get("confidence", 0.5))
    needs_research = bool(data.get("needs_research", False))
    reasoning = data.get("reasoning", "")

    return RouterResult(
        show_type=show_type,
        confidence=confidence,
        reasoning=reasoning,
        needs_research=needs_research,
    )


class ShowRouter:
    """Classifies a brief into a show type via a single LLM call."""

    def __init__(self, client: DeepSeekClient):
        self._client = client

    def classify(self, brief: str) -> RouterResult:
        """Classify a brief and return a RouterResult."""
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": ROUTER_USER_TEMPLATE.format(brief=brief)},
        ]
        raw = self._client.chat(messages, temperature=0)
        return parse_router_response(raw)
