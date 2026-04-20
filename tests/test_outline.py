"""Tests for the DirectorOutline — versioned outline with change classification.

Tests:
    1. Create outline with initial sections
    2. Snapshot generation
    3. Small change classification (key_beats, duration)
    4. Major change classification (speaker_assignments, section removal/addition)
    5. Version incrementing on update
    6. Changed section detection
    7. get_section lookup
    8. to_broadcast_text formatting

Run with:
    cd /Users/bagheera/repos/vox_romana
    python -m pytest tests/test_outline.py -v
"""

import pytest

from vox_romana.core.editing.models import OutlineSection, OutlineSnapshot
from vox_romana.core.editing.outline import DirectorOutline


def make_sections() -> list[OutlineSection]:
    """Standard 3-section outline for testing."""
    return [
        OutlineSection(
            id="intro",
            title="Opening Segment",
            expected_duration="5 min",
            key_beats=["hook", "set scene"],
            speaker_assignments=["Alex", "Jordan"],
            acceptance_criteria="Engaging opening",
        ),
        OutlineSection(
            id="main",
            title="Main Discussion",
            expected_duration="15 min",
            key_beats=["thesis", "counterpoint", "evidence"],
            speaker_assignments=["Alex", "Jordan"],
            acceptance_criteria="Substantive debate",
        ),
        OutlineSection(
            id="closing",
            title="Closing Statements",
            expected_duration="5 min",
            key_beats=["summary", "final appeal"],
            speaker_assignments=["Alex", "Jordan"],
            acceptance_criteria="Clear conclusions",
        ),
    ]


class TestCreation:
    def test_initial_version_is_one(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        assert outline.version == 1

    def test_outline_id_assigned(self):
        outline = DirectorOutline(make_sections(), outline_id="custom_id")
        assert outline.outline_id == "custom_id"

    def test_auto_id_generated(self):
        outline = DirectorOutline(make_sections())
        assert outline.outline_id != ""
        assert len(outline.outline_id) == 8

    def test_sections_preserved(self):
        sections = make_sections()
        outline = DirectorOutline(sections, outline_id="test")
        assert len(outline.sections) == 3
        assert outline.sections[0].id == "intro"

    def test_sections_are_copy(self):
        sections = make_sections()
        outline = DirectorOutline(sections, outline_id="test")
        sections.append(OutlineSection(id="extra", title="Extra"))
        assert len(outline.sections) == 3  # unaffected


class TestSnapshot:
    def test_snapshot_fields(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        snap = outline.snapshot(summary="Initial plan")
        assert snap.outline_id == "test"
        assert snap.version == 1
        assert snap.summary == "Initial plan"
        assert snap.timestamp != ""
        assert len(snap.sections) == 3

    def test_snapshot_default_summary(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        snap = outline.snapshot()
        assert "v1" in snap.summary

    def test_to_broadcast_text(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        snap = outline.snapshot(summary="Test broadcast")
        text = snap.to_broadcast_text()
        assert "OUTLINE UPDATE (v1)" in text
        assert "Test broadcast" in text
        assert "[intro]" in text
        assert "Opening Segment" in text
        assert "Alex" in text


class TestSmallChanges:
    def test_key_beats_change_is_small(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        updated = make_sections()
        updated[0].key_beats = ["hook", "set scene", "tease main topic"]
        snap, change_type = outline.update(updated, "Added a beat")
        assert change_type == "small"

    def test_duration_change_is_small(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        updated = make_sections()
        updated[1].expected_duration = "20 min"
        snap, change_type = outline.update(updated, "Extended main")
        assert change_type == "small"

    def test_acceptance_criteria_change_is_small(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        updated = make_sections()
        updated[2].acceptance_criteria = "Strong emotional finish"
        snap, change_type = outline.update(updated, "Tightened criteria")
        assert change_type == "small"


class TestMajorChanges:
    def test_speaker_assignment_change_is_major(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        updated = make_sections()
        updated[0].speaker_assignments = ["Alex", "Morgan"]  # Jordan → Morgan
        snap, change_type = outline.update(updated, "Swapped speaker")
        assert change_type == "major"

    def test_section_removal_is_major(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        updated = make_sections()[:2]  # removed closing
        snap, change_type = outline.update(updated, "Removed closing")
        assert change_type == "major"

    def test_section_addition_is_major(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        updated = make_sections()
        updated.append(OutlineSection(id="bonus", title="Bonus Round"))
        snap, change_type = outline.update(updated, "Added bonus round")
        assert change_type == "major"

    def test_section_merge_detected(self):
        """Replacing two sections with one should be major."""
        outline = DirectorOutline(make_sections(), outline_id="test")
        merged = [
            make_sections()[0],
            OutlineSection(
                id="combined",
                title="Main + Closing",
                speaker_assignments=["Alex", "Jordan"],
            ),
        ]
        snap, change_type = outline.update(merged, "Merged sections")
        assert change_type == "major"


class TestVersioning:
    def test_version_increments_on_update(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        assert outline.version == 1
        outline.update(make_sections(), "No real change")
        assert outline.version == 2
        outline.update(make_sections(), "Another update")
        assert outline.version == 3

    def test_snapshot_version_matches(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        outline.update(make_sections(), "Update")
        snap = outline.snapshot()
        assert snap.version == outline.version


class TestChangedSections:
    def test_detects_specific_changed_section(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        updated = make_sections()
        updated[1].key_beats = ["new_beat"]
        snap, _ = outline.update(updated, "Changed main beats")
        assert "main" in snap.changed_sections

    def test_unchanged_sections_not_flagged(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        updated = make_sections()
        updated[1].key_beats = ["new_beat"]
        snap, _ = outline.update(updated, "Changed main beats")
        assert "intro" not in snap.changed_sections
        assert "closing" not in snap.changed_sections

    def test_removed_section_flagged(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        updated = make_sections()[:2]
        snap, _ = outline.update(updated, "Removed closing")
        assert "closing" in snap.changed_sections


class TestGetSection:
    def test_finds_existing(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        s = outline.get_section("main")
        assert s is not None
        assert s.title == "Main Discussion"

    def test_returns_none_for_missing(self):
        outline = DirectorOutline(make_sections(), outline_id="test")
        assert outline.get_section("nonexistent") is None
