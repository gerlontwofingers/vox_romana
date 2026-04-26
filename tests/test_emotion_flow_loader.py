"""Tests for vox_romana.emotion_flow.loader — FlexibleLoader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vox_romana.emotion_flow.loader import FlexibleLoader
from vox_romana.emotion_flow.models import SpeechSegment


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_txt(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


def _make_vox_json(events: list[dict], layout: str = "flat") -> dict:
    """Build a minimal vox_romana JSON with the given events."""
    if layout == "flat":
        return {"events": events}
    elif layout == "segments":
        return {"segments": [{"events": events}]}
    elif layout == "chunks":
        return {"chunks": [{"events": events}]}
    raise ValueError(f"Unknown layout: {layout}")


def _utterance(speaker: str, text: str) -> dict:
    return {
        "event_kind": "utterance",
        "speaker_name": speaker,
        "text": text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Text file loading
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadTextFile:
    def test_splits_on_double_newline(self, tmp_path):
        p = _write_txt(
            tmp_path, "speech.txt",
            "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.",
        )
        segments = FlexibleLoader.load(p)
        assert len(segments) == 3

    def test_speech_id_is_filename_stem(self, tmp_path):
        p = _write_txt(
            tmp_path, "lincoln_1865.txt",
            "Four score and seven years ago.\n\n"
            "Now we are engaged in a great civil war.",
        )
        segments = FlexibleLoader.load(p)
        assert all(s.speech_id == "lincoln_1865" for s in segments)

    def test_segment_index_is_sequential(self, tmp_path):
        p = _write_txt(
            tmp_path, "s.txt",
            "First segment content here.\n\n"
            "Second segment content here.\n\n"
            "Third segment content here.",
        )
        segs = FlexibleLoader.load(p)
        assert [s.segment_index for s in segs] == [0, 1, 2]

    def test_short_paragraphs_filtered(self, tmp_path):
        p = _write_txt(
            tmp_path, "s.txt",
            "Hi.\n\nThis is a substantive paragraph with enough words.",
        )
        segs = FlexibleLoader.load(p, min_chars=20)
        assert len(segs) == 1
        assert "substantive" in segs[0].text

    def test_source_lang_empty_before_translation(self, tmp_path):
        p = _write_txt(
            tmp_path, "s.txt",
            "This is a proper paragraph with enough characters to pass.",
        )
        segs = FlexibleLoader.load(p)
        assert all(s.source_lang == "" for s in segs)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FlexibleLoader.load(tmp_path / "nonexistent.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Directory loading
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadDirectory:
    def test_loads_multiple_files(self, tmp_path):
        _write_txt(
            tmp_path, "speech_a.txt",
            "First speech first paragraph.\n\nFirst speech second paragraph.",
        )
        _write_txt(tmp_path, "speech_b.txt", "Second speech only paragraph here.")
        segs = FlexibleLoader.load(tmp_path)
        assert len(segs) >= 3

    def test_speech_ids_differ_across_files(self, tmp_path):
        _write_txt(tmp_path, "a.txt", "Content for file a paragraph one here.")
        _write_txt(tmp_path, "b.txt", "Content for file b paragraph one here.")
        segs = FlexibleLoader.load(tmp_path)
        ids = {s.speech_id for s in segs}
        assert "a" in ids
        assert "b" in ids

    def test_global_index_is_sequential(self, tmp_path):
        _write_txt(tmp_path, "a.txt", "Para one.\n\nPara two here for length.")
        _write_txt(tmp_path, "b.txt", "Para three for length check.")
        segs = FlexibleLoader.load(tmp_path, min_chars=15)
        indices = [s.segment_index for s in segs]
        assert indices == list(range(len(segs)))

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No .txt files"):
            FlexibleLoader.load(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# vox_romana JSON loading
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadVoxRomanaJson:
    def test_flat_layout_single_speaker(self, tmp_path):
        data = _make_vox_json([
            _utterance("Churchill", "We shall fight on the beaches with great determination."),
            _utterance("Churchill", "We shall never surrender under any circumstances whatsoever."),
        ])
        p = tmp_path / "speech.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        segs = FlexibleLoader.load(p)  # no leader needed — only one speaker
        assert len(segs) == 2

    def test_segments_layout(self, tmp_path):
        data = _make_vox_json(
            [_utterance("Alice", "This is a long enough segment for Alice here.")],
            layout="segments",
        )
        p = tmp_path / "pod.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        segs = FlexibleLoader.load(p, leader="Alice")
        assert len(segs) == 1

    def test_chunks_layout(self, tmp_path):
        data = _make_vox_json(
            [_utterance("Bob", "This is Bob's debate segment with sufficient length.")],
            layout="chunks",
        )
        p = tmp_path / "debate.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        segs = FlexibleLoader.load(p, leader="bob")  # case-insensitive
        assert len(segs) == 1

    def test_leader_filter_case_insensitive(self, tmp_path):
        data = _make_vox_json([
            _utterance("Churchill", "We shall fight on the beaches with great determination here."),
            _utterance("Roosevelt", "We shall discuss lend-lease policy in great detail today."),
        ])
        p = tmp_path / "debate.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        segs = FlexibleLoader.load(p, leader="CHURCHILL")
        assert len(segs) == 1
        assert "beaches" in segs[0].text

    def test_multiple_speakers_no_leader_raises(self, tmp_path):
        data = _make_vox_json([
            _utterance("Alice", "First speaker says something quite long here."),
            _utterance("Bob", "Second speaker says something quite long here."),
        ])
        p = tmp_path / "debate.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="multiple speakers"):
            FlexibleLoader.load(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FlexibleLoader.load(tmp_path / "nope.json")


# ─────────────────────────────────────────────────────────────────────────────
# Real vox_romana JSON (integration, skipped if not present)
# ─────────────────────────────────────────────────────────────────────────────

_REAL_DEBATE = (
    Path(__file__).parent.parent
    / "output/text/debates/tealc_v_ronin_01_stargate_20260405_163843.json"
)


@pytest.mark.skipif(not _REAL_DEBATE.exists(), reason="Real debate JSON not present")
class TestRealDebateJson:
    def test_loads_tealc(self):
        segs = FlexibleLoader.load(_REAL_DEBATE, leader="Teal")
        assert len(segs) > 5
        assert all(isinstance(s, SpeechSegment) for s in segs)
