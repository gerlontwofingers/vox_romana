"""Lecture show — single-voice didactic educational content.

A lecture is a SCRIPTED show type with one professor/teacher voice
delivering a structured, pedagogical talk on a topic. Unlike audiobooks
(narrative prose) or podcasts (conversational), lectures use explicit
teaching beats: introduce → define → exemplify → recap.

Module layout:
    config.py     → LectureConfig + ProfessorConfig + LectureSection
    models.py     → LectureEvent + LectureSection + MergedLecture
    prompts.py    → ARCHITECT, PROFESSOR, SECTION, DIRECTOR system prompts
    bootstrap.py  → Architect (brief → outline) + Professor profiler
    conductor.py  → Section-by-section delivery loop
    serializer.py → Pretty-printed transcript writer
"""

from .bootstrap import LectureBootstrap
from .conductor import LectureConductor, LectureConductorResult
from .config import (
    LectureConfig,
    LectureDirectorBrief,
    LectureSection,
    ProfessorConfig,
)
from .models import (
    LectureEvent,
    LectureSection as LectureSectionModel,
    MergedLecture,
)
from .serializer import LectureSerializer

__all__ = [
    "LectureBootstrap",
    "LectureConductor",
    "LectureConductorResult",
    "LectureConfig",
    "LectureDirectorBrief",
    "LectureSection",
    "ProfessorConfig",
    "LectureEvent",
    "LectureSectionModel",
    "MergedLecture",
    "LectureSerializer",
]
