"""Tests for vox_romana.analysis.corpus — CorpusLoader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vox_romana.analysis.corpus import CorpusLoader, _extract_events
from vox_romana.analysis.models import SpeakerCorpus

_REAL_DEBATE_JSON = (
    Path(__file__).parent.parent
    / "output/text/debates/tealc_v_ronin_01_stargate_20260405_163843.json"
)
_HAS_REAL_DEBATE = _REAL_DEBATE_JSON.exists()


# ─────────────────────────────────────────────────────────────────────────────
# SpeakerCorpus model
# ─────────────────────────────────────────────────────────────────────────────

class TestSpeakerCorpus:
    def test_properties(self):
        corpus = SpeakerCorpus(name="Alice", utterances=["Hello world", "Foo bar baz"])
        assert corpus.word_count == 5
        assert corpus.utterance_count == 2
        assert corpus.full_text == "Hello world Foo bar baz"

    def test_empty_utterances_raises(self):
        with pytest.raises(ValueError, match="at least one utterance"):
            SpeakerCorpus(name="Alice", utterances=[])


# ─────────────────────────────────────────────────────────────────────────────
# from_vox_romana_json
# ─────────────────────────────────────────────────────────────────────────────

class TestFromVoxRomanaJson:
    @pytest.mark.skipif(not _HAS_REAL_DEBATE, reason="Real debate JSON not present")
    def test_loads_real_debate_json(self):
        corpora = CorpusLoader.from_vox_romana_json(_REAL_DEBATE_JSON)
        assert isinstance(corpora, dict)
        assert len(corpora) >= 2
        for name, corpus in corpora.items():
            assert isinstance(corpus, SpeakerCorpus)
            assert corpus.name == name
            assert len(corpus.utterances) > 0
            for utt in corpus.utterances:
                assert isinstance(utt, str)
                assert utt.strip() != ""

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CorpusLoader.from_vox_romana_json(tmp_path / "nonexistent.json")

    def test_exclude_speakers(self, tmp_path):
        data = _make_chunks_json([
            ("Alice", "participant", "utterance", "I believe in policy A which is good"),
            ("Bob", "participant", "utterance", "I disagree completely with that position"),
            ("Alice", "participant", "utterance", "Let me explain my second argument here"),
        ])
        f = tmp_path / "debate.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        corpora = CorpusLoader.from_vox_romana_json(f, exclude_speakers={"Bob"})
        assert "Bob" not in corpora
        assert "Alice" in corpora

    def test_min_words_filters_short_utterances(self, tmp_path):
        data = _make_chunks_json([
            ("Alice", "participant", "utterance", "OK"),
            ("Alice", "participant", "utterance", "Yes I agree"),
            ("Bob", "participant", "utterance", "This is a longer sentence indeed"),
        ])
        f = tmp_path / "debate.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        corpora = CorpusLoader.from_vox_romana_json(f, min_words=3)
        assert "Alice" in corpora
        assert all(len(u.split()) >= 3 for u in corpora["Alice"].utterances)

    def test_flat_events_layout(self, tmp_path):
        data = {
            "title": "Flat test",
            "events": [
                _make_event(1, "Alice", "participant", "utterance",
                            "This is the first point I want to make"),
                _make_event(2, "Bob", "participant", "utterance",
                            "And here is my rebuttal to that claim"),
            ],
        }
        f = tmp_path / "flat.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        corpora = CorpusLoader.from_vox_romana_json(f)
        assert "Alice" in corpora
        assert "Bob" in corpora

    def test_segments_layout(self, tmp_path):
        data = {
            "show_id": "news_01",
            "title": "Test News",
            "segments": [
                {
                    "segment_id": "s1",
                    "events": [
                        _make_event(1, "Anchor", "anchor", "utterance",
                                    "Good evening and welcome to the show tonight"),
                        _make_event(2, "Reporter", "reporter", "utterance",
                                    "Tonight we cover the main story developing here"),
                    ],
                }
            ],
        }
        f = tmp_path / "news.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        corpora = CorpusLoader.from_vox_romana_json(f)
        assert "Anchor" in corpora
        assert "Reporter" in corpora

    def test_no_qualifying_utterances_raises(self, tmp_path):
        data = _make_chunks_json([
            ("Alice", "participant", "break", "..."),
        ])
        f = tmp_path / "debate.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="No qualifying utterances"):
            CorpusLoader.from_vox_romana_json(f)

    def test_metadata_includes_source_file(self, tmp_path):
        data = _make_chunks_json([
            ("Alice", "participant", "utterance", "A complete sentence with enough words"),
        ])
        f = tmp_path / "debate.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        corpora = CorpusLoader.from_vox_romana_json(f)
        assert corpora["Alice"].metadata["source_file"] == str(f)


# ─────────────────────────────────────────────────────────────────────────────
# from_text_file
# ─────────────────────────────────────────────────────────────────────────────

class TestFromTextFile:
    def test_loads_text_file(self, tmp_path):
        content = (
            "This is the first line of speech.\n"
            "This is the second line here.\n"
            "Short."
        )
        f = tmp_path / "speech.txt"
        f.write_text(content, encoding="utf-8")

        corpus = CorpusLoader.from_text_file(f)
        assert corpus.name == "speech"
        assert len(corpus.utterances) == 2
        assert "This is the first line of speech." in corpus.utterances

    def test_speaker_name_defaults_to_stem(self, tmp_path):
        f = tmp_path / "candidate_a.txt"
        f.write_text("This is a longer utterance for testing purposes here.\n",
                     encoding="utf-8")
        corpus = CorpusLoader.from_text_file(f)
        assert corpus.name == "candidate_a"

    def test_speaker_name_override(self, tmp_path):
        f = tmp_path / "speech.txt"
        f.write_text("This is a test sentence with enough words.\n", encoding="utf-8")
        corpus = CorpusLoader.from_text_file(f, speaker_name="Candidate A")
        assert corpus.name == "Candidate A"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CorpusLoader.from_text_file(tmp_path / "nope.txt")

    def test_min_words_filter(self, tmp_path):
        content = "Hello.\nThis is a full sentence with enough words in it.\nBye."
        f = tmp_path / "speech.txt"
        f.write_text(content, encoding="utf-8")

        corpus = CorpusLoader.from_text_file(f, min_words=5)
        assert len(corpus.utterances) == 1
        assert "full sentence" in corpus.utterances[0]

    def test_no_qualifying_lines_raises(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("Hi.\nOK.\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No qualifying lines"):
            CorpusLoader.from_text_file(f, min_words=10)


# ─────────────────────────────────────────────────────────────────────────────
# from_ingest_path
# ─────────────────────────────────────────────────────────────────────────────

class TestFromIngestPath:
    def test_loads_text_via_ingest(self, tmp_path):
        content = "\n".join(
            f"This is utterance number {i} and it has enough words to qualify."
            for i in range(10)
        )
        f = tmp_path / "corpus.txt"
        f.write_text(content, encoding="utf-8")

        corpus = CorpusLoader.from_ingest_path(f, speaker_name="Author")
        assert corpus.name == "Author"
        assert len(corpus.utterances) >= 1

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CorpusLoader.from_ingest_path(tmp_path / "nope.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# _extract_events helper
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractEvents:
    def test_chunks_layout(self):
        data = {
            "chunks": [
                {"events": [_make_event(1, "A", "participant", "utterance", "text one")]},
                {"events": [_make_event(2, "B", "participant", "utterance", "text two")]},
            ]
        }
        events = _extract_events(data)
        assert len(events) == 2
        assert events[0]["global_order"] == 1
        assert events[1]["global_order"] == 2

    def test_segments_layout(self):
        data = {
            "segments": [
                {"events": [_make_event(1, "A", "anchor", "utterance", "segment one")]},
            ]
        }
        events = _extract_events(data)
        assert len(events) == 1

    def test_flat_layout(self):
        data = {
            "events": [
                _make_event(1, "A", "participant", "utterance", "flat event here"),
            ]
        }
        events = _extract_events(data)
        assert len(events) == 1

    def test_deduplication_by_global_order(self):
        ev = _make_event(1, "A", "participant", "utterance", "duplicate event text")
        data = {
            "chunks": [{"events": [ev]}],
            "events": [ev],
        }
        events = _extract_events(data)
        assert len(events) == 1

    def test_ordering_by_global_order(self):
        data = {
            "chunks": [
                {"events": [_make_event(3, "A", "participant", "utterance", "third")]},
                {"events": [_make_event(1, "B", "participant", "utterance", "first")]},
                {"events": [_make_event(2, "A", "participant", "utterance", "second")]},
            ]
        }
        events = _extract_events(data)
        assert [e["global_order"] for e in events] == [1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_event(
    global_order: int,
    speaker_name: str,
    speaker_role: str,
    event_kind: str,
    text: str,
) -> dict:
    return {
        "global_order": global_order,
        "local_order": global_order,
        "speaker_name": speaker_name,
        "speaker_role": speaker_role,
        "event_kind": event_kind,
        "text": text,
        "tone": [],
        "nonverbal": [],
        "interruption": False,
        "overlap_with_previous": False,
        "target": None,
    }


def _make_chunks_json(events: list[tuple]) -> dict:
    return {
        "title": "Test Debate",
        "debate_id": "test_01",
        "chunks": [
            {
                "chunk_order": 1,
                "is_final_chunk": True,
                "events": [
                    _make_event(i + 1, name, role, kind, text)
                    for i, (name, role, kind, text) in enumerate(events)
                ],
            }
        ],
    }
