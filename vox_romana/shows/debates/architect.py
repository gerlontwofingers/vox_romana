"""The Architect agent — turns a brief into a debate skeleton.

If the bootstrap chain is an assembly line, the Architect is the first station.
The human hands it a napkin sketch ("two economists argue about UBI") and the
Architect hands back a full blueprint: title, format, participants with sides,
moderator, audience, setting, tone, and target length.

The Architect does NOT fill in personalities or dramatic arcs. Those are the
Profile Builder's and Director's jobs, respectively. The Architect just lays
out the cast, the stage, and the ground rules.

SINGLE-SHOT DESIGN:
    Unlike the DeepSeekAgent class (which maintains conversation history for
    multi-turn dialogues), the Architect makes ONE call to the LLM. It sends
    a system prompt + user message, gets JSON back, and parses it. No state,
    no memory, no follow-ups.

    Why? Because the Architect's job is a pure function:
        brief + reference_material → DebateConfig skeleton

    If it screws up, we can retry from scratch. There's nothing to "remember."

TESTABILITY:
    The `parse_response()` method is deliberately separated from `run()` so
    you can unit-test the JSON→DebateConfig parsing without hitting the LLM.
    Feed it a canned JSON string = same result every time.

EXAMPLE:
    from vox_romana.core import DeepSeekClient
    from vox_romana.shows.debates.architect import DebateArchitect

    client = DeepSeekClient()
    architect = DebateArchitect(client)

    config = architect.run(
        brief="Two economists debate universal basic income. "
              "One is a progressive academic, the other a libertarian think-tanker.",
        reference_material="Finland's 2017 UBI pilot... Stockton CA SEED program...",
    )

    print(config.title)          # "The UBI Question: Promise or Pipe Dream?"
    print(config.participants)   # [ParticipantProfile(...), ParticipantProfile(...)]
    print(config.debate_format)  # "public_forum"
"""

from __future__ import annotations

import json

from ...core.client import DeepSeekClient
from .config import (
    AudienceConfig,
    DebateConfig,
    ModeratorConfig,
    ParticipantProfile,
    ShowHostConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# The system prompt is the "job description" we give the LLM. It explains what
# role it's playing and exactly what format we expect back. Be specific about
# the JSON schema or you'll spend hours debugging malformed responses.
# ─────────────────────────────────────────────────────────────────────────────

ARCHITECT_SYSTEM_PROMPT = """\
You are the Architect for a debate generation system. Your job is to take a
human's brief (a description of the debate they want) and produce a structured
debate configuration.

You will receive:
  1. A brief — the user's natural language description of the debate.
  2. Optionally, reference material — factual source text about the topic.

You must produce a JSON object with EXACTLY this schema. Do NOT include any
text outside the JSON block. No markdown, no commentary, just raw JSON.

{
  "title": "string — catchy title for the debate",
  "topic": "string — one-sentence distillation of the core question",
  "debate_format": "string — one of: public_forum, crossfire, oxford, panel, town_hall, roundtable",
  "setting": "string — where this debate is happening (e.g., 'university lecture hall')",
  "tone": "string — overall mood (e.g., 'academic but accessible', 'cable news energy')",
  "target_turns": "integer — total speaking turns to plan for (20–60 typical)",
  "participants": [
    {
      "name": "string — full name (invented or real, matching the brief)",
      "side": "string — which position they argue (e.g., 'pro-UBI', 'anti-regulation')",
      "role": "participant"
    }
  ],
  "moderator": {
    "name": "string — moderator name",
    "mode": "string — active, passive, or none",
    "style": "string — moderation style description"
  },
  "audience": {
    "present": "boolean",
    "participation": "boolean — can the audience interject?",
    "size_description": "string — e.g., 'packed town hall, standing room only'"
  },
  "show_host": {
    "name": "string — host's name (a character, e.g., 'Marcus Webb' or 'The Narrator')",
    "role": "string — 'narrator' (third-person, documentary style) or 'emcee' (first-person, podcast host style)",
    "style": "string — hosting style description, e.g., 'Warm but authoritative, uses vivid scene-setting language'"
  }
}

RULES:
- Usually include 2-4 participants. More than 4 is a panel, not a debate.
- If the brief mentions specific people, use their real names.
- If the brief is vague about format, pick the best fit and explain via setting/tone.
- moderator can be null if the format doesn't need one (e.g., crossfire).
- audience can have present=false for studio debates.
- show_host adds narrative structure (intro, transitions, outro). Include one
  for produced/polished shows. Use "narrator" for formal or documentary tones,
  "emcee" for casual/podcast vibes. Can be null if the brief explicitly asks
  for no host, but PREFER including one — it makes the show sound professional.
- target_turns: aim for ~1 turn per 30-45 seconds of final audio.
  A 20-minute debate ≈ 30-40 turns. IMPORTANT: scale with available content.
  With 2 participants and NO reference material, keep to 20-30 turns — each
  speaker only has ~10-15 talking points generated from general knowledge.
  With reference material or 3+ participants, go higher (30-50). A speaker
  with no research backing needs about 1.5-2 turns per distinct argument
  angle before repetition sets in. When in doubt, err on the shorter side —
  a tight 20-turn debate beats a bloated 45-turn one with recycled arguments.
- ALWAYS respond with valid JSON. Nothing else.
"""


class DebateArchitect:
    """Converts a brief into a skeleton DebateConfig.

    This is a single-purpose tool: give it a brief, get back a config with
    the cast, format, and ground rules filled in. Personalities and dramatic
    arcs are NOT its responsibility.

    Attributes:
        _client: The DeepSeekClient to use for LLM calls. Accepts it via
                 constructor so tests can inject a mock.
    """

    def __init__(self, client: DeepSeekClient) -> None:
        # We take the client as a parameter rather than creating our own.
        # This is "dependency injection" — a fancy way of saying "let someone
        # else decide which phone we use to call the LLM." In production,
        # the Bootstrap passes in the real client. In tests, it passes a mock.
        self._client = client

    def run(
        self,
        brief: str,
        *,
        reference_material: str | None = None,
        fact_check: str = "off",
    ) -> DebateConfig:
        """Call the LLM and return a skeleton DebateConfig.

        Args:
            brief:              Natural language description of the debate.
            reference_material: Optional factual source text to ground the output.
            fact_check:         Fact-check mode to set on the resulting config.

        Returns:
            A DebateConfig with cast, format, and ground rules filled in.
            Participant profiles will be stubs (name + side only).
            director_brief will be None (that's the Director's job).

        Raises:
            ValueError: If the LLM returns garbage that can't be parsed.
        """
        # ── Build the user message ────────────────────────────────────
        # The user message contains the brief and (optionally) reference
        # material. We label them clearly so the LLM knows what's what.
        user_parts = [f"BRIEF:\n{brief}"]
        if reference_material:
            user_parts.append(f"\nREFERENCE MATERIAL:\n{reference_material}")

        user_message = "\n".join(user_parts)

        # ── Call the LLM ──────────────────────────────────────────────
        # Single-shot. System prompt defines the role + output schema.
        # User message provides the specific input.
        raw_response = self._client.chat(
            messages=[
                {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )

        # ── Parse the JSON response into a DebateConfig ──────────────
        return self.parse_response(
            raw_response,
            brief=brief,
            reference_material=reference_material,
            fact_check=fact_check,
        )

    def parse_response(
        self,
        raw_json: str,
        *,
        brief: str = "",
        reference_material: str | None = None,
        fact_check: str = "off",
    ) -> DebateConfig:
        """Parse the LLM's JSON output into a DebateConfig.

        This is deliberately a separate method so it can be unit-tested
        with canned JSON strings. No LLM call needed.

        Args:
            raw_json:           The raw string from the LLM (should be JSON).
            brief:              Original brief (stored on the config for provenance).
            reference_material: Original reference material.
            fact_check:         Fact-check mode.

        Returns:
            A DebateConfig with all architecture-level fields populated.

        Raises:
            ValueError: If the JSON is malformed or missing required fields.
        """
        # ── Step 1: Extract JSON from the response ────────────────────
        # Sometimes LLMs wrap JSON in ```json ... ``` markdown fences
        # even when told not to. Strip that if present.
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            # Remove the first line (```json) and last line (```)
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Architect produced invalid JSON. First 200 chars: "
                f"{raw_json[:200]!r}\nParse error: {e}"
            ) from e

        # ── Step 2: Build ParticipantProfile stubs ────────────────────
        # At this stage, participants are just names + sides. The Profile
        # Builder will flesh them out with personalities later.
        participants = []
        for p in data.get("participants", []):
            participants.append(
                ParticipantProfile(
                    name=p.get("name", ""),
                    side=p.get("side"),
                    role=p.get("role", "participant"),
                )
            )

        # ── Step 3: Build moderator config (if present) ──────────────
        mod_data = data.get("moderator")
        moderator = None
        if mod_data:
            moderator = ModeratorConfig(
                name=mod_data.get("name", ""),
                mode=mod_data.get("mode", "active"),
                style=mod_data.get("style"),
            )

        # ── Step 4: Build audience config (if present) ────────────────
        aud_data = data.get("audience")
        audience = None
        if aud_data:
            audience = AudienceConfig(
                present=aud_data.get("present", False),
                participation=aud_data.get("participation", False),
                size_description=aud_data.get("size_description"),
            )

        # ── Step 5: Build show host config (if present) ──────────────
        # The show host is the narrative voice that wraps the debate in
        # structure — intro, transitions, outro. The Architect decides
        # whether to include one based on the brief's vibe.
        host_data = data.get("show_host")
        show_host = None
        if host_data:
            show_host = ShowHostConfig(
                name=host_data.get("name", ""),
                role=host_data.get("role", "narrator"),
                style=host_data.get("style"),
            )

        # ── Step 6: Assemble the DebateConfig ────────────────────────
        return DebateConfig(
            brief=brief,
            reference_material=reference_material,
            fact_check=fact_check,
            title=data.get("title"),
            topic=data.get("topic"),
            debate_format=data.get("debate_format"),
            setting=data.get("setting"),
            tone=data.get("tone"),
            target_turns=data.get("target_turns"),
            participants=participants,
            moderator=moderator,
            audience=audience,
            show_host=show_host,
        )
