"""The Director Brief agent — plans the dramatic arc for a debate.

The Director is a film director, not a participant. It doesn't debate — it
shapes the EXPERIENCE of the debate. Where should the tension escalate?
When should someone make a surprising concession? How does the audience
mood shift? When do we land the plane?

Think of it like this:
    - The Architect builds the stage and assigns the roles.
    - The Profile Builder gives each actor a personality.
    - The Director writes the dramatic arc — the invisible hand that
      guides the conversation from "polite introductions" to "heated
      rebuttals" to "reluctant mutual respect."

During generation, the DebateConductor consults the DirectorBrief every
`check_interval` turns and nudges the conversation toward the planned arc.
It's not rigid — the agents improvise — but the Director provides structure
so the debate doesn't meander or stall.

TESTABILITY:
    Same pattern as the Architect and Profile Builder: `parse_response()`
    is separated from `run()` for unit testing with canned JSON.

EXAMPLE:
    from vox_romana.core import DeepSeekClient
    from vox_romana.shows.debates.director import DebateDirector

    client = DeepSeekClient()
    director = DebateDirector(client)

    brief = director.run(
        topic="Should the US adopt universal basic income?",
        participants=["Dr. Sarah Chen (pro-UBI)", "James Hartley (anti-UBI)"],
        debate_format="public_forum",
        tone="academic but accessible",
        target_turns=35,
    )

    print(brief.dramatic_arc)
    # "Three-act structure. Act 1 (turns 1-10): establish positions..."
    print(brief.mood_progression)
    # ["academic", "engaged", "heated", "personal", "reconciliatory"]
"""

from __future__ import annotations

import json

from ...core.client import DeepSeekClient
from .config import DirectorBrief


# ─────────────────────────────────────────────────────────────────────────────
# The Director's system prompt. It needs to know the debate's cast, format,
# and tone so it can design an appropriate arc. A cable-news crossfire gets
# a different arc than an academic roundtable.
# ─────────────────────────────────────────────────────────────────────────────

DIRECTOR_SYSTEM_PROMPT = """\
You are the Director for a debate generation system. Your job is to plan the
dramatic arc — the emotional journey of the debate from opening to closing.

You will receive details about the debate: topic, participants, format, tone,
and target length. Using these, design a dramatic arc that creates compelling
audio content.

You must produce a JSON object with EXACTLY this schema. Do NOT include any
text outside the JSON block.

{
  "dramatic_arc": "string — detailed description of the narrative structure. \
Reference specific turn ranges. Example: 'Three-act structure. Act 1 (turns \
1-10): polite opening statements, establishing credibility. Act 2 (turns \
11-25): escalation, specific rebuttals, personal anecdotes, one interruption \
around turn 18. Act 3 (turns 26-35): peak tension at turn 28, then gradual \
de-escalation, closing statements with unexpected common ground.'",

  "mood_progression": ["array of strings — ordered emotional waypoints. \
5-8 entries. Example: ['academic', 'engaged', 'heated', 'personal', \
'conciliatory']. The Director uses these as beacons during generation."],

  "key_moments": ["array of strings — 3-6 planned dramatic beats the Director \
should try to engineer. Be specific about what happens and roughly when. \
Example: 'Around turn 15, moderator challenges the frontrunner with a \
surprising statistic', 'Near turn 25, one participant makes an unexpected \
personal revelation'"],

  "closing_strategy": "string — how to land the plane. Describe the last \
5-8 turns. Example: 'Each participant gets a 2-turn closing statement. \
Moderator summarizes the 3 key points of disagreement. Final moment: one \
participant acknowledges the other made a good point.'"
}

RULES:
- The arc must match the tone. An "academic" debate builds slowly. A
  "cable news" debate starts hot and stays hot.
- Key moments should be ACHIEVABLE by the agents, not absurdly specific.
  "Someone makes a concession" is good. "Dr. Chen quotes Keynes from
  memory while the audience gasps" is too prescriptive.
- mood_progression should be 5-8 entries, roughly mapping to the debate's
  timeline from start to finish.
- dramatic_arc should reference turn numbers relative to target_turns.
  It MUST include TOPICAL PHASES — don't just describe emotional shifts, plan
  what SUB-TOPICS or ANGLES the debate should explore in each phase. Great
  debates NARROW from broad claims to specific examples and edge cases.
  Example: "Act 1: broad principles. Act 2: specific policies and data.
  Act 3: implementation challenges and personal stakes."
- key_moments should include at least one TOPICAL PIVOT — a moment where
  the conversation shifts to a completely new angle or sub-topic.
- closing_strategy is critical. A debate that just stops feels terrible.
  Plan the landing. Each participant gets ONE closing statement, not multiple.
- ALWAYS respond with valid JSON. Nothing else.
"""


class DebateDirector:
    """Plans the dramatic arc for a debate.

    Takes debate metadata (topic, format, participants, tone, target length)
    and produces a DirectorBrief with the dramatic arc, mood progression,
    key moments, and closing strategy.
    """

    def __init__(self, client: DeepSeekClient) -> None:
        self._client = client

    def run(
        self,
        *,
        topic: str,
        participants: list[str],
        debate_format: str = "public_forum",
        tone: str = "balanced",
        target_turns: int = 35,
    ) -> DirectorBrief:
        """Call the LLM and return a DirectorBrief.

        Args:
            topic:          The debate topic.
            participants:   List of "Name (side)" strings for context.
            debate_format:  The debate format (e.g., "public_forum").
            tone:           The overall tone/mood target.
            target_turns:   How many turns the debate should run.

        Returns:
            A DirectorBrief with dramatic arc and pacing info.

        Raises:
            ValueError: If the LLM returns unparseable JSON.
        """
        # ── Build user message with debate details ────────────────────
        participant_list = "\n".join(f"  - {p}" for p in participants)
        user_message = (
            f"TOPIC: {topic}\n"
            f"FORMAT: {debate_format}\n"
            f"TONE: {tone}\n"
            f"TARGET TURNS: {target_turns}\n"
            f"PARTICIPANTS:\n{participant_list}"
        )

        raw_response = self._client.chat(
            messages=[
                {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )

        return self.parse_response(raw_response)

    def parse_response(self, raw_json: str) -> DirectorBrief:
        """Parse the LLM's JSON into a DirectorBrief.

        Testable without the LLM — pass a canned JSON string.

        Args:
            raw_json: The raw string from the LLM (should be JSON).

        Returns:
            A DirectorBrief with all fields populated.

        Raises:
            ValueError: If JSON is malformed.
        """
        # ── Strip markdown fences ─────────────────────────────────────
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Director produced invalid JSON. First 200 chars: "
                f"{raw_json[:200]!r}\nParse error: {e}"
            ) from e

        return DirectorBrief(
            dramatic_arc=data.get("dramatic_arc", ""),
            mood_progression=data.get("mood_progression", []),
            key_moments=data.get("key_moments", []),
            closing_strategy=data.get("closing_strategy"),
        )
