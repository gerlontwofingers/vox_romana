"""Tests for the vox_romana.publish package.

Tests use small inline fixtures (so they pass on a fresh checkout where
``output/`` is empty) with optional opportunistic checks against real
artefacts when present locally.

PDF tests are skipped when ``weasyprint`` isn't importable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vox_romana.publish import (
    PDF_AVAILABLE,
    DebateFormatter,
    NewsFormatter,
    PodcastFormatter,
    AudiobookFormatter,
    detect_show_type,
    format_document,
    publish_document,
    render_txt,
)
from vox_romana.publish.formatter import _norm_list  # type: ignore
from vox_romana.publish.models import FormattedDocument


# ─────────────────────────────────────────────────────────────────────────────
# Inline minimal fixtures matching the real serializer shapes
# ─────────────────────────────────────────────────────────────────────────────

def _ev(
    speaker: str,
    text: str,
    role: str = "participant",
    kind: str = "utterance",
    nonverbal: list[str] | None = None,
    order: int = 1,
) -> dict:
    return {
        "global_order": order,
        "local_order": order,
        "speaker_role": role,
        "speaker_name": speaker,
        "target": None,
        "text": text,
        "tone": [],
        "volume": "moderate",
        "pace": "medium",
        "pause_before": 0.0,
        "pause_after": 0.0,
        "interruption": False,
        "nonverbal": nonverbal or [],
        "event_kind": kind,
        "overlap_with_previous": False,
        "completion_status": "complete",
        "segment_id": None,
        "raw_fields": {},
    }


def _debate_fixture() -> dict:
    events = [
        _ev("Moderator Smith", "Welcome to tonight's debate.",
            role="moderator", order=1),
        _ev("Alice", "I believe the answer is yes.", order=2,
            nonverbal=["leans forward"]),
        _ev("Bob", "I respectfully disagree.", order=3),
        _ev("Alice", "Then explain why.", order=4),
    ]
    chunk = {
        "template_id": "tpl_01",
        "debate_id": "deb_01",
        "title": "Test Debate",
        "chunk_order": 1,
        "is_final_chunk": True,
        "participants": ["Alice", "Bob"],
        "moderators": ["Moderator Smith"],
        "hosts": [],
        "audience_present": False,
        "audience_participation": False,
        "topic": "An important topic",
        "debate_format": "Oxford",
        "setting": "A grand auditorium",
        "moderator_mode": "active",
        "continuity_notes": [],
        "metadata": {},
        "events": events,
    }
    return {
        "template_id": "tpl_01",
        "debate_id": "deb_01",
        "title": "Test Debate",
        "chunks": [chunk],
        "events": events,
        "parser_names": [],
        "participants": ["Alice", "Bob"],
        "moderators": ["Moderator Smith"],
        "hosts": [],
        "audience_present": False,
        "audience_participation": False,
        "topic": "An important topic",
        "debate_format": "Oxford",
        "setting": "A grand auditorium",
        "moderator_mode": "active",
        "continuity_notes": [],
        "source_count": 0,
    }


def _news_fixture() -> dict:
    seg1 = {
        "segment_id": "seg_01",
        "segment_order": 0,
        "headline": "Council approves new park",
        "segment_type": "anchor_read",
        "anchor_name": "Jane Doe",
        "tone": "neutral",
        "events": [
            _ev("Jane Doe", "Good evening. Tonight, the council voted...",
                role="anchor", order=1),
            _ev("Jane Doe", "The vote was unanimous.",
                role="anchor", order=2),
        ],
    }
    seg2 = {
        "segment_id": "seg_02",
        "segment_order": 1,
        "headline": "Local sports update",
        "segment_type": "package",
        "anchor_name": "John Roe",
        "tone": "upbeat",
        "events": [
            _ev("John Roe", "Now to sports.", role="anchor", order=3),
            _ev("Field Reporter", "I'm here at the stadium.",
                role="reporter", order=4),
        ],
    }
    return {
        "show_id": "news_01",
        "title": "Evening News",
        "segments": [seg1, seg2],
        "events": seg1["events"] + seg2["events"],
        "anchors": ["Jane Doe", "John Roe"],
        "hosts": [],
        "guests": [],
        "topic": "Local news",
        "tone": "neutral",
        "segment_count": 2,
        "voice_casting": {},
    }


def _podcast_fixture() -> dict:
    seg = {
        "segment_id": "p_seg_01",
        "segment_order": 1,
        "segment_type": "banter",
        "topic": "Coffee preferences",
        "events": [
            _ev("Alex", "I love a good espresso.", role="host", order=1),
            _ev("Sam", "Espresso? Pour-over all the way.",
                role="host", order=2,
                nonverbal=["laughs"]),
        ],
    }
    return {
        "podcast_id": "pod_01",
        "title": "Two Hosts Talk",
        "show_name": "Talk Show",
        "topic": "Coffee",
        "tone": "casual",
        "hosts": ["Alex", "Sam"],
        "segments": [seg],
        "events": seg["events"],
        "voice_casting": {},
        "voice_hints": {},
        "tts_styles": {},
    }


def _audiobook_fixture() -> dict:
    chapter = {
        "chapter_id": "ch_01",
        "chapter_order": 1,
        "title": "The Beginning",
        "tone": "ominous",
        "narrator_name": "Narrator",
        "events": [
            _ev("Narrator", "It was a dark and stormy night.",
                role="narrator", order=1),
            _ev("Frank", "Get out!", order=2),
            _ev("Narrator", "He turned and fled into the rain.",
                role="narrator", order=3),
        ],
    }
    return {
        "audiobook_id": "ab_01",
        "title": "A Test Tale",
        "chapters": [chapter],
        "events": chapter["events"],
        "narrator": "Narrator",
        "characters": ["Frank"],
        "genre": "Mystery",
        "tone": "ominous",
        "chapter_count": 1,
        "voice_casting": {},
        "voice_hints": {},
        "tts_styles": {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectShowType:
    def test_debate(self):
        assert detect_show_type(_debate_fixture()) == "debate"

    def test_news(self):
        assert detect_show_type(_news_fixture()) == "news"

    def test_podcast(self):
        assert detect_show_type(_podcast_fixture()) == "podcast"

    def test_audiobook(self):
        assert detect_show_type(_audiobook_fixture()) == "audiobook"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            detect_show_type({"foo": "bar"})


# ─────────────────────────────────────────────────────────────────────────────
# DebateFormatter
# ─────────────────────────────────────────────────────────────────────────────

class TestDebateFormatter:
    def test_basic_shape(self):
        doc = DebateFormatter().format(_debate_fixture())
        assert isinstance(doc, FormattedDocument)
        assert doc.show_type == "debate"
        assert doc.title == "Test Debate"
        assert len(doc.sections) == 1

    def test_front_matter_includes_topic_and_cast(self):
        doc = DebateFormatter().format(_debate_fixture())
        joined = " | ".join(l.text for l in doc.front_matter)
        assert "An important topic" in joined
        assert "Alice" in joined
        assert "Bob" in joined
        assert "Moderator Smith" in joined

    def test_speakers_uppercased(self):
        doc = DebateFormatter().format(_debate_fixture())
        speaker_lines = [
            l.text for sec in doc.sections for l in sec.lines
            if l.kind == "speaker"
        ]
        assert any(s.startswith("ALICE") for s in speaker_lines)
        assert any(s.startswith("BOB") for s in speaker_lines)
        # All speaker labels end with a colon.
        assert all(s.endswith(":") for s in speaker_lines)

    def test_dialogue_present(self):
        doc = DebateFormatter().format(_debate_fixture())
        body = " ".join(
            l.text for sec in doc.sections for l in sec.lines
            if l.kind == "dialogue"
        )
        assert "I believe the answer is yes." in body

    def test_nonverbal_renders_as_direction(self):
        doc = DebateFormatter().format(_debate_fixture())
        directions = [
            l.text for sec in doc.sections for l in sec.lines
            if l.kind == "direction"
        ]
        assert any("leans forward" in d for d in directions)

    def test_multi_chunk_creates_acts(self):
        data = _debate_fixture()
        chunk2 = dict(data["chunks"][0])
        chunk2["chunk_order"] = 2
        chunk2["events"] = [
            _ev("Alice", "Round two.", order=10),
        ]
        data["chunks"] = [data["chunks"][0], chunk2]
        doc = DebateFormatter().format(data)
        titles = [s.title for s in doc.sections]
        assert "ACT 1" in titles
        assert "ACT 2" in titles


# ─────────────────────────────────────────────────────────────────────────────
# NewsFormatter
# ─────────────────────────────────────────────────────────────────────────────

class TestNewsFormatter:
    def test_basic_shape(self):
        doc = NewsFormatter().format(_news_fixture())
        assert doc.show_type == "news"
        assert len(doc.sections) == 2

    def test_segment_titles_have_slug_and_headline(self):
        doc = NewsFormatter().format(_news_fixture())
        titles = [s.title for s in doc.sections]
        assert any("SLUG 01" in t for t in titles)
        assert any("COUNCIL APPROVES NEW PARK" in t for t in titles)

    def test_segment_type_label_in_metadata(self):
        doc = NewsFormatter().format(_news_fixture())
        first = doc.sections[0]
        meta = " | ".join(l.text for l in first.lines if l.kind == "metadata")
        # anchor_read → ANCHOR ON CAM
        assert "ANCHOR ON CAM" in meta
        assert "Jane Doe" in meta

    def test_role_labels_in_speaker_lines(self):
        doc = NewsFormatter().format(_news_fixture())
        speakers = [
            l.text for sec in doc.sections for l in sec.lines
            if l.kind == "speaker"
        ]
        assert any("ANCHOR" in s for s in speakers)
        assert any("REPORTER" in s for s in speakers)

    def test_uses_per_segment_events_not_top_level(self):
        # If we accidentally rendered top-level events, segment 1 would
        # contain segment 2's reporter line.  Verify isolation.
        doc = NewsFormatter().format(_news_fixture())
        first_dialogue = " ".join(
            l.text for l in doc.sections[0].lines if l.kind == "dialogue"
        )
        assert "stadium" not in first_dialogue


# ─────────────────────────────────────────────────────────────────────────────
# PodcastFormatter
# ─────────────────────────────────────────────────────────────────────────────

class TestPodcastFormatter:
    def test_basic_shape(self):
        doc = PodcastFormatter().format(_podcast_fixture())
        assert doc.show_type == "podcast"
        assert doc.title == "Two Hosts Talk"
        assert len(doc.sections) == 1

    def test_speakers_with_colon(self):
        doc = PodcastFormatter().format(_podcast_fixture())
        speakers = [
            l.text for sec in doc.sections for l in sec.lines
            if l.kind == "speaker"
        ]
        assert "Alex:" in speakers
        assert "Sam:" in speakers

    def test_nonverbal_inline_brackets(self):
        doc = PodcastFormatter().format(_podcast_fixture())
        dialogue = " ".join(
            l.text for sec in doc.sections for l in sec.lines
            if l.kind == "dialogue"
        )
        # 'laughs' → [LAUGHTER]
        assert "[LAUGHTER]" in dialogue


# ─────────────────────────────────────────────────────────────────────────────
# AudiobookFormatter
# ─────────────────────────────────────────────────────────────────────────────

class TestAudiobookFormatter:
    def test_basic_shape(self):
        doc = AudiobookFormatter().format(_audiobook_fixture())
        assert doc.show_type == "audiobook"
        assert len(doc.sections) == 1
        assert doc.sections[0].title.startswith("Chapter 1")
        assert "The Beginning" in doc.sections[0].title

    def test_narration_renders_as_prose(self):
        doc = AudiobookFormatter().format(_audiobook_fixture())
        narration = [
            l.text for l in doc.sections[0].lines if l.kind == "narration"
        ]
        assert "It was a dark and stormy night." in narration

    def test_dialogue_quoted_with_attribution(self):
        doc = AudiobookFormatter().format(_audiobook_fixture())
        dialogue = [
            l.text for l in doc.sections[0].lines if l.kind == "dialogue"
        ]
        # Terminal '!' is preserved; attribution follows without a comma.
        assert any('"Get out!"' in d and "said Frank" in d for d in dialogue)


# ─────────────────────────────────────────────────────────────────────────────
# format_document dispatcher
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatDocument:
    @pytest.mark.parametrize(
        "fx,kind",
        [
            (_debate_fixture(),    "debate"),
            (_news_fixture(),      "news"),
            (_podcast_fixture(),   "podcast"),
            (_audiobook_fixture(), "audiobook"),
        ],
    )
    def test_dispatches_correct_formatter(self, fx, kind):
        doc = format_document(fx)
        assert doc.show_type == kind


# ─────────────────────────────────────────────────────────────────────────────
# render_txt
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderTxt:
    def test_writes_file(self, tmp_path):
        doc = format_document(_debate_fixture())
        out = tmp_path / "debate.txt"
        render_txt(doc, out)
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "TEST DEBATE" in text  # title uppercased
        assert "ALICE:" in text
        assert "I believe the answer is yes." in text

    def test_news_txt_has_slug(self, tmp_path):
        doc = format_document(_news_fixture())
        out = tmp_path / "news.txt"
        render_txt(doc, out)
        text = out.read_text(encoding="utf-8")
        assert "SLUG 01" in text
        assert "ANCHOR" in text

    def test_audiobook_txt_no_speaker_labels_for_narrator(self, tmp_path):
        doc = format_document(_audiobook_fixture())
        out = tmp_path / "book.txt"
        render_txt(doc, out)
        text = out.read_text(encoding="utf-8")
        # Narrator name should NOT appear as a speaker label.
        assert "NARRATOR:" not in text
        assert "It was a dark and stormy night." in text
        # Dialogue uses quotation + said attribution.
        assert '"Get out!"' in text
        assert "said Frank" in text

    def test_collapses_blank_runs(self, tmp_path):
        doc = format_document(_debate_fixture())
        out = tmp_path / "d.txt"
        render_txt(doc, out)
        text = out.read_text(encoding="utf-8")
        assert "\n\n\n\n" not in text


# ─────────────────────────────────────────────────────────────────────────────
# publish_document
# ─────────────────────────────────────────────────────────────────────────────

class TestPublishDocument:
    def test_txt_only(self, tmp_path):
        src = tmp_path / "in.json"
        src.write_text(json.dumps(_debate_fixture()), encoding="utf-8")
        result = publish_document(src, formats=("txt",), dest_dir=tmp_path)
        assert result.txt_path is not None
        assert result.pdf_path is None
        assert result.show_type == "debate"
        assert result.sections == 1
        assert result.txt_path.exists()

    def test_unknown_format_raises(self, tmp_path):
        src = tmp_path / "in.json"
        src.write_text(json.dumps(_news_fixture()), encoding="utf-8")
        with pytest.raises(ValueError):
            publish_document(src, formats=("md",), dest_dir=tmp_path)

    def test_pdf_skipped_warning_when_unavailable(self, tmp_path, monkeypatch):
        src = tmp_path / "in.json"
        src.write_text(json.dumps(_news_fixture()), encoding="utf-8")
        # Force the "PDF unavailable" path even if weasyprint is installed.
        import vox_romana.publish.outputter as r
        monkeypatch.setattr(r, "PDF_AVAILABLE", False)
        result = r.publish_document(
            src, formats=("txt", "pdf"), dest_dir=tmp_path,
        )
        assert result.pdf_path is None
        assert any("PDF skipped" in w for w in result.warnings)
        assert result.txt_path is not None

    @pytest.mark.skipif(not PDF_AVAILABLE,
                        reason="weasyprint and/or jinja2 not installed")
    def test_pdf_real(self, tmp_path):
        src = tmp_path / "in.json"
        src.write_text(json.dumps(_debate_fixture()), encoding="utf-8")
        result = publish_document(
            src, formats=("pdf",), dest_dir=tmp_path,
        )
        assert result.pdf_path is not None
        assert result.pdf_path.exists()
        assert result.pdf_path.read_bytes()[:4] == b"%PDF"


# ─────────────────────────────────────────────────────────────────────────────
# Opportunistic real-fixture smoke tests (skipped if files absent)
# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_DEBATE = (
    _REPO_ROOT / "output" / "text" / "debates"
    / "tealc_v_ronin_01_stargate_20260405_163843.json"
)
_REAL_NEWS = (
    _REPO_ROOT / "output" / "text" / "news"
    / "news_places_test_01_local_news_test_20260406_003942.json"
)


@pytest.mark.skipif(not _REAL_DEBATE.exists(),
                    reason="Real debate fixture not present locally")
def test_real_debate_fixture_renders(tmp_path):
    data = json.loads(_REAL_DEBATE.read_text(encoding="utf-8"))
    doc = format_document(data)
    assert doc.show_type == "debate"
    assert len(doc.sections) >= 1
    out = tmp_path / "real_debate.txt"
    render_txt(doc, out)
    assert out.exists()
    assert out.stat().st_size > 200


@pytest.mark.skipif(not _REAL_NEWS.exists(),
                    reason="Real news fixture not present locally")
def test_real_news_fixture_renders(tmp_path):
    data = json.loads(_REAL_NEWS.read_text(encoding="utf-8"))
    doc = format_document(data)
    assert doc.show_type == "news"
    assert len(doc.sections) >= 1
    out = tmp_path / "real_news.txt"
    render_txt(doc, out)
    assert out.exists()
    assert out.stat().st_size > 200


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestNormList:
    def test_none(self):
        assert _norm_list(None) == []

    def test_string(self):
        assert _norm_list("a") == ["a"]
        assert _norm_list("  ") == []

    def test_list(self):
        assert _norm_list(["a", " b ", ""]) == ["a", "b"]
