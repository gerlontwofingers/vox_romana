"""AudienceProfile — structured representation of a series' target audience.

Stored as JSON in the ``series.audience_profile_json`` column.  When set on a
series, the profile is appended to the SERIES CONTEXT block that every
Architect, Profile Builder, and Director reads before processing a brief.

This lets showrunners tune vocabulary, assumed prior knowledge, and tone
once — at the series level — rather than spelling it out in every prompt.

Usage:

    from vox_romana.core.audience import AudienceProfile

    profile = AudienceProfile(
        age_range="25-45",
        education_level="graduate",
        vocabulary_level="educated",
        preferred_tone="rigorous but accessible",
        prior_knowledge="Familiar with basic economics; no formal stats background.",
    )
    json_str = profile.to_json()          # store in DB
    p2 = AudienceProfile.from_json(json_str)  # round-trip
    print(profile.to_prompt_text())       # inject into Architect prompt
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


# Allowed vocabulary levels (ordered from simplest to most technical).
VOCABULARY_LEVELS = ("plain", "conversational", "educated", "expert")
EDUCATION_LEVELS = ("high-school", "undergraduate", "graduate", "expert")


@dataclass
class AudienceProfile:
    """Who the show is for.

    All fields are optional except ``vocabulary_level`` which has a
    sensible default ("conversational") so the Architect always receives
    at least that signal.

    Attributes:
        age_range:         Free-form age band, e.g. "18-35" or "40+".
        education_level:   Broadest educational background assumed.
                           One of: high-school, undergraduate, graduate, expert.
        vocabulary_level:  Expected vocabulary complexity.  Controls word choice
                           in the generated dialogue.
                           One of: plain, conversational, educated, expert.
        preferred_tone:    Free-form description, e.g. "warm, skeptical".
        prior_knowledge:   What the listener is assumed to already know.
    """

    age_range: str | None = None
    education_level: str | None = None
    vocabulary_level: str = "conversational"
    preferred_tone: str | None = None
    prior_knowledge: str | None = None

    # ── Serialization ────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize to a JSON string for database storage."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> AudienceProfile:
        """Deserialize from a JSON string.

        Unknown keys in the stored JSON are silently ignored, so old
        records stay readable after adding new fields.
        """
        data = json.loads(s)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    # ── Prompt injection ─────────────────────────────────────────────

    def to_prompt_text(self) -> str:
        """Format as a human-readable block for insertion into LLM prompts.

        The output is intentionally terse — it occupies a handful of
        lines in the SERIES CONTEXT header so it doesn't crowd out the
        brief itself.
        """
        lines: list[str] = []
        if self.age_range:
            lines.append(f"Age range: {self.age_range}")
        if self.education_level:
            lines.append(f"Education: {self.education_level}")
        lines.append(f"Vocabulary: {self.vocabulary_level}")
        if self.preferred_tone:
            lines.append(f"Tone preference: {self.preferred_tone}")
        if self.prior_knowledge:
            lines.append(f"Prior knowledge: {self.prior_knowledge}")
        return "\n".join(lines)
