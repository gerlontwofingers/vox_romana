"""Director Outline — versioned outline with change classification.

The Director creates an initial outline at broadcast start and may make
small adjustments during the run. This module tracks versions, classifies
changes as small or major, and produces snapshots for broadcasting.

CHANGE CLASSIFICATION (from the design spec):
  SMALL (Director may make freely, then broadcast):
    - Adjusting key_beats within an existing section
    - Changing expected_duration
    - Reordering beats within a section
    - Adding a minor transitional beat

  MAJOR (must be flagged to Conductor, who may warn or veto):
    - Altering speaker_assignments
    - Removing a section entirely
    - Adding a new section not in the original
    - Merging two sections into one
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from .models import OutlineSection, OutlineSnapshot


class DirectorOutline:
    """A versioned, mutable show outline with change classification.

    The Director holds one of these for the entire broadcast. Any update
    increments the version and produces a snapshot suitable for broadcasting.
    """

    def __init__(
        self,
        initial_sections: list[OutlineSection],
        outline_id: str | None = None,
    ) -> None:
        self._outline_id = outline_id or str(uuid.uuid4())[:8]
        self._sections = list(initial_sections)
        self._original_section_ids = frozenset(s.id for s in initial_sections)
        self._original_assignments = {
            s.id: list(s.speaker_assignments) for s in initial_sections
        }
        self._version = 1

    @property
    def outline_id(self) -> str:
        return self._outline_id

    @property
    def version(self) -> int:
        return self._version

    @property
    def sections(self) -> list[OutlineSection]:
        return list(self._sections)

    def snapshot(
        self,
        summary: str = "",
        changed_sections: list[str] | None = None,
    ) -> OutlineSnapshot:
        """Create a snapshot of the current outline state."""
        return OutlineSnapshot(
            outline_id=self._outline_id,
            version=self._version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary=summary or f"Outline v{self._version}",
            sections=list(self._sections),
            changed_sections=changed_sections or [],
        )

    def update(
        self,
        updated_sections: list[OutlineSection],
        summary: str = "",
    ) -> tuple[OutlineSnapshot, str]:
        """Update the outline. Returns (snapshot, change_type).

        change_type is "small" or "major". The caller (Conductor) should
        check change_type and decide whether to allow or veto.
        """
        change_type = self._classify_change(updated_sections)
        changed_ids = self._find_changed_sections(updated_sections)

        self._sections = list(updated_sections)
        self._version += 1

        snap = self.snapshot(
            summary=summary or f"Outline updated (v{self._version})",
            changed_sections=changed_ids,
        )
        return snap, change_type

    def get_section(self, section_id: str) -> OutlineSection | None:
        """Look up a section by ID."""
        for s in self._sections:
            if s.id == section_id:
                return s
        return None

    def _classify_change(self, updated_sections: list[OutlineSection]) -> str:
        """Classify the change as 'small' or 'major'."""
        new_ids = frozenset(s.id for s in updated_sections)

        # Section removed?
        if self._original_section_ids - new_ids:
            return "major"

        # Section added that wasn't in original?
        if new_ids - self._original_section_ids:
            return "major"

        # Section count changed (merge)?
        if len(updated_sections) != len(self._sections):
            return "major"

        # Speaker assignments changed from original?
        for section in updated_sections:
            original = self._original_assignments.get(section.id, [])
            if list(section.speaker_assignments) != original:
                return "major"

        return "small"

    def _find_changed_sections(
        self, updated_sections: list[OutlineSection]
    ) -> list[str]:
        """Find which section IDs have changed."""
        current_map = {s.id: s for s in self._sections}
        changed: list[str] = []

        for section in updated_sections:
            old = current_map.get(section.id)
            if old is None:
                changed.append(section.id)
            elif (
                section.key_beats != old.key_beats
                or section.expected_duration != old.expected_duration
                or section.speaker_assignments != old.speaker_assignments
                or section.acceptance_criteria != old.acceptance_criteria
                or section.title != old.title
            ):
                changed.append(section.id)

        # Also flag any removed sections
        new_ids = {s.id for s in updated_sections}
        for s in self._sections:
            if s.id not in new_ids:
                changed.append(s.id)

        return changed
