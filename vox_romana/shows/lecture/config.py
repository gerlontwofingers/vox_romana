"""Lecture configuration models.

A lecture is a single-voice didactic show. The professor walks listeners
through a topic using structured pedagogy — introduce, define, exemplify,
recap. Sections are typed (concept | case_study | exercise | recap) so the
conductor can sequence pedagogical beats appropriately.

Config hierarchy:
    ContentConfig (core)
        └── LectureConfig
                ├── ProfessorConfig    — the single voice
                ├── LectureSection[]   — ordered section outline
                └── LectureDirectorBrief — pacing/coverage arc
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.config import ContentConfig, VoiceProfile


SECTION_TYPES = ("concept", "case_study", "exercise", "recap", "intro", "outro")


@dataclass
class ProfessorConfig(VoiceProfile):
    """The single lecturer voice for the show."""

    name: str = ""
    title: str | None = None  # e.g., "Professor of Cognitive Science"
    style: str | None = None  # e.g., "Warm, methodical, fond of analogies"
    expertise: list[str] = field(default_factory=list)


@dataclass
class LectureSection:
    """Configuration for a single lecture section.

    Sections are typed so the conductor knows which pedagogical beats to
    request. A typical lecture sequences: intro → concept → case_study →
    concept → exercise → recap → outro. The architect picks the mix.
    """

    title: str = ""
    section_type: str = "concept"  # one of SECTION_TYPES
    learning_objective: str | None = None  # What the listener should grasp
    key_points: list[str] = field(default_factory=list)  # Ordered teaching beats
    examples: list[str] = field(default_factory=list)  # Concrete illustrations
    duration_hint: str = "medium"  # "short", "medium", "long"
    target_beats: int = 1  # Number of LLM generation beats for this section


@dataclass
class LectureDirectorBrief:
    """Pacing and pedagogical-arc plan for the lecture.

    Attributes:
        thesis:           One-sentence statement of what the lecture argues
                          or teaches. The "throughline."
        progression:      Ordered list of conceptual milestones the lecture
                          should hit (e.g. ["foundations", "complications",
                          "synthesis"]).
        pedagogical_notes: Director's high-level notes on teaching style —
                          how often to use analogies, when to recap, etc.
        closing_strategy: How to land the lecture — "open question",
                          "definitive synthesis", "call to action".
        check_interval:   How many *sections* between Director check-ins.
                          Default: every 3 sections.
    """

    thesis: str = ""
    progression: list[str] = field(default_factory=list)
    pedagogical_notes: str | None = None
    closing_strategy: str | None = None
    check_interval: int = 3


@dataclass
class LectureConfig(ContentConfig):
    """Full configuration for lecture generation.

    Extends ContentConfig with lecture-specific fields: professor, sections,
    subject area, and pedagogical brief.
    """

    content_type: str = "lecture"
    flexibility_score: float = 0.15  # Even more scripted than audiobook
    topic: str | None = None  # Subject of the lecture
    subject_area: str | None = None  # Discipline (e.g., "philosophy", "ML")
    tone: str | None = None  # Overall tone
    setting: str | None = None  # E.g., "intimate seminar", "auditorium"
    professor: ProfessorConfig | None = None
    sections: list[LectureSection] = field(default_factory=list)
    director_brief: LectureDirectorBrief | None = None

    def validate(self) -> list[str]:
        """Return a list of validation problems (empty = valid)."""
        problems = super().validate()
        if not self.sections:
            problems.append("LectureConfig requires at least one section.")
        if not self.professor:
            problems.append("LectureConfig requires a professor.")
        elif not self.professor.name:
            problems.append("Professor must have a name.")
        for i, sec in enumerate(self.sections):
            if sec.section_type not in SECTION_TYPES:
                problems.append(
                    f"Section {i} has invalid section_type "
                    f"{sec.section_type!r}; expected one of {SECTION_TYPES}."
                )
        return problems
