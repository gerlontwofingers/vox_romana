"""The Profile Builder agent — turns participant stubs into full personalities.

If the Architect lays out the cast list, the Profile Builder is the character
designer. It takes a bare-bones participant stub (name + side) and creates a
rich, nuanced personality profile: how they talk, what triggers them, how they
evolve during the debate, and what factual claims they'll use.

THIS IS WHERE DISTINCTIVENESS LIVES.

Without the Profile Builder, every debater sounds like the same polite LLM
wearing a different name tag. With it, Dr. Sarah Chen interrupts with
data-heavy rebuttals while Mayor Ortiz tells personal stories from his
constituency. The voice_fingerprint is the single most important field for
making debates feel authentic.

PARALLELISM:
    Profile building is embarrassingly parallel. If the debate has 3
    participants, we can build all 3 profiles simultaneously — they're
    independent. The Bootstrap orchestrator handles this parallelism.

TESTABILITY:
    Like the Architect, the `parse_response()` method is separated from
    `run()` for easy unit testing with canned JSON.

EXAMPLE:
    from vox_romana.core import DeepSeekClient
    from vox_romana.shows.debates.profile_builder import DebateProfileBuilder
    from vox_romana.shows.debates.config import ParticipantProfile

    client = DeepSeekClient()
    builder = DebateProfileBuilder(client)

    stub = ParticipantProfile(name="Dr. Sarah Chen", side="pro-UBI")
    full_profile = builder.run(
        stub,
        topic="Should the US adopt universal basic income?",
        reference_material="Finland's 2017 pilot showed...",
    )

    print(full_profile.voice_fingerprint)
    # "Starts sentences with hedging phrases ('The data suggests...',
    #  'It's worth noting that...'). Builds to emphatic conclusions.
    #  Favors passive constructions when citing studies."
"""

from __future__ import annotations

import json

from ...core.client import DeepSeekClient
from .config import ParticipantProfile


# ─────────────────────────────────────────────────────────────────────────────
# The system prompt tells the LLM it's a character psychologist. We ask for
# specific personality dimensions that map 1:1 to ParticipantProfile fields.
# ─────────────────────────────────────────────────────────────────────────────

PROFILE_BUILDER_SYSTEM_PROMPT = """\
You are a Character Psychologist for a debate generation system. Your job is
to take a participant stub (name + side) and flesh it out into a rich, nuanced
personality profile that will make this character sound authentic and distinct.

You will receive:
  1. A participant name and the side they're arguing.
  2. The debate topic.
  3. Optionally, reference material — factual source text about the topic.

You must produce a JSON object with EXACTLY this schema. Do NOT include any
text outside the JSON block.

{
  "voice_gender": "string — 'male' or 'female'. The biological vocal gender \
of this character for text-to-speech voice assignment. Base this on who the \
character IS, not how they argue. For real people and fictional characters, \
use their canonical gender.",

  "voice_hint": "string — a short natural-language casting note describing \
the ideal SOUND of this character's voice. Think of it as the note a casting \
director tapes to the audition sheet. Focus on vocal qualities: accent, age \
range, energy, warmth, authority, texture. Examples: 'warm British baritone, \
late 40s, professorial', 'young California surfer energy, upbeat', 'sharp \
New York newsroom voice, 50s, no-nonsense'. This is NOT about what they say \
— it's about how they SOUND.",

  "tts_style": "string — a persistent text-to-speech style prefix that will \
be prepended to EVERY utterance this character delivers. This instructs the \
voice engine on ongoing delivery style. Keep it to 1-2 sentences. Examples: \
'Speak with quiet authority and measured pacing, like a senior professor \
delivering a keynote.', 'Deliver with punchy cable-news energy, slightly \
breathless, like breaking news.'  This persists across all turns — it's the \
character's baseline vocal personality, not a per-line emotion cue.",

  "voice_fingerprint": "string — how this person TALKS, not what they BELIEVE. \
Rhetorical habits, sentence patterns, favorite phrases, verbal tics, \
preferred sentence length, use of questions vs statements, formality level.",

  "behavioral_triggers": "string — how they react to specific situations: \
being interrupted, being challenged with data, when the audience laughs, \
when they're losing an argument, when attacked personally.",

  "emotional_arc": "string — how their demeanor should evolve over the debate. \
Start state → middle state → end state. Be specific about turns or timing.",

  "key_positions": ["array of strings — 8-15 DISTINCT argument angles, NOT just \
top-level positions. For each major stance, include multiple sub-arguments: \
specific examples, implementation details, edge cases, historical parallels, \
moral/philosophical angles, and the strongest counter to the opposing side. \
Order from strongest to weakest. Each entry should be a DIFFERENT angle — \
not a restatement. A speaker with 5 broad positions should have 2-3 specific \
angles per position. Think 'enough material for 20+ minutes of conversation.'"],

  "factual_claims": ["array of strings — specific, verifiable claims they should \
cite. MUST be grounded in the reference material if provided. If no reference \
material, leave this array empty rather than inventing claims. Each claim \
should include its source (study name, year, statistic)."],

  "tangent_topics": ["array of strings — 3-6 personality-driven digressions this \
person naturally reaches for. These are NOT main arguments — they're the \
personal anecdotes, analogies from other fields, rhetorical questions, \
historical parallels, or thought experiments this character gravitates toward \
based on their background and personality. Examples: an economist might tangent \
into game theory puzzles; a politician into constituent stories; an academic \
into obscure historical precedents. These keep the conversation feeling natural \
and human when core arguments have been covered."],

  "reference_corpus": ["array of strings — 5-12 example quotes showing how this \
person would phrase things. These are FEW-SHOT examples for the generation \
agent. Match the voice fingerprint. Include a MIX: some forceful arguments, \
some concessions, some tangential asides, some reactions to being challenged. \
Make them feel natural, not rehearsed."]
}

RULES:
- voice_fingerprint is the MOST IMPORTANT field. Put real effort into making
  it specific and distinctive. Generic descriptions ("speaks clearly and
  confidently") are useless. Describe quirks, habits, patterns.
- For real public figures, base the voice on their actual speaking style.
- key_positions should have DEPTH, not just breadth. A 2-participant debate
  may run 30-40 turns (15-20 per speaker). With only 5 talking points,
  the speaker runs dry by turn 10. Provide 8-15 distinct angles: main
  stances, sub-arguments, concrete examples, philosophical angles, edge
  cases, historical parallels, and direct counters to the opposing side.
- factual_claims MUST come from reference_material. If none is provided,
  return an empty array. NEVER invent statistics or studies.
- tangent_topics are personality-driven diversions — the kind of thing
  this person would naturally bring up based on who they are. They make
  conversations feel human and prevent the speaker from robotically
  cycling through arguments. Make them character-specific.
- reference_corpus quotes should demonstrate the voice_fingerprint in action.
  They're examples of HOW this person phrases things, not just WHAT they say.
  Include variety: forceful, concessive, tangential, and reactive quotes.
- ALWAYS respond with valid JSON. Nothing else.
"""


class DebateProfileBuilder:
    """Fleshes out a participant stub into a full personality profile.

    Takes a ParticipantProfile with just name + side, calls the LLM once,
    and populates all the personality fields (voice_fingerprint, behavioral
    triggers, emotional_arc, etc.).

    The original stub is not mutated. A new, fully populated
    ParticipantProfile is returned.
    """

    def __init__(self, client: DeepSeekClient) -> None:
        self._client = client

    def run(
        self,
        stub: ParticipantProfile,
        *,
        topic: str = "",
        reference_material: str | None = None,
    ) -> ParticipantProfile:
        """Call the LLM and return a fully fleshed-out profile.

        Args:
            stub:               A ParticipantProfile with at least name and side.
            topic:              The debate topic (gives the LLM context).
            reference_material: Factual source text for grounding claims.

        Returns:
            A new ParticipantProfile with all personality fields populated.
            The name, side, and role from the stub are preserved.

        Raises:
            ValueError: If the LLM returns unparseable JSON.
        """
        # ── Build the user message ────────────────────────────────────
        user_parts = [
            f"PARTICIPANT: {stub.name}",
            f"SIDE: {stub.side or 'unspecified'}",
            f"ROLE: {stub.role}",
            f"TOPIC: {topic}",
        ]
        if reference_material:
            user_parts.append(f"\nREFERENCE MATERIAL:\n{reference_material}")

        user_message = "\n".join(user_parts)

        # ── Single-shot LLM call ──────────────────────────────────────
        raw_response = self._client.chat(
            messages=[
                {"role": "system", "content": PROFILE_BUILDER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )

        return self.parse_response(raw_response, stub=stub)

    def parse_response(
        self,
        raw_json: str,
        *,
        stub: ParticipantProfile,
    ) -> ParticipantProfile:
        """Parse the LLM's JSON into a fully populated ParticipantProfile.

        Testable without the LLM — just pass a canned JSON string.

        Args:
            raw_json: The raw string from the LLM (should be JSON).
            stub:     The original stub whose name/side/role we preserve.

        Returns:
            A new ParticipantProfile with personality fields filled in.

        Raises:
            ValueError: If JSON is malformed.
        """
        # ── Strip markdown fences if present ──────────────────────────
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Profile Builder produced invalid JSON. First 200 chars: "
                f"{raw_json[:200]!r}\nParse error: {e}"
            ) from e

        # ── Build a new profile, preserving the stub's identity fields ─
        # We create a fresh object rather than mutating the stub, because
        # the caller might want to retry or compare before/after.
        return ParticipantProfile(
            name=stub.name,
            side=stub.side,
            role=stub.role,
            voice_gender=data.get("voice_gender"),
            voice_hint=data.get("voice_hint"),
            tts_style=data.get("tts_style"),
            voice_fingerprint=data.get("voice_fingerprint"),
            behavioral_triggers=data.get("behavioral_triggers"),
            emotional_arc=data.get("emotional_arc"),
            key_positions=data.get("key_positions", []),
            factual_claims=data.get("factual_claims", []),
            reference_corpus=data.get("reference_corpus", []),
            tangent_topics=data.get("tangent_topics", []),
        )
