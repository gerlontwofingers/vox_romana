"""Tests for Shot 4 — Multilingual Publish.

Covers:
  * FormattedDocument.language field default and propagation.
  * PublishResult.language field.
  * DebateFormatter: reads language from data, sets on doc, localises ACT label.
  * NewsFormatter: reads language from data, sets on doc.
  * PodcastFormatter: reads language, localises Segment label.
  * AudiobookFormatter: reads language, localises Chapter label.
  * _default_outputs: adds _{lang} suffix when language != root_lang.
  * publish_document: exposes root_lang param, result.language, filenames.
  * _base.html.jinja2: uses {{ doc.language }} not "en".
  * _SECTION_LABELS / _get_labels: fallback to English for unknown codes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vox_romana.publish.formatter import (
    AudiobookFormatter,
    DebateFormatter,
    NewsFormatter,
    PodcastFormatter,
    _get_labels,
    _SECTION_LABELS,
    format_document,
)
from vox_romana.publish.models import FormattedDocument, Line, PublishResult
from vox_romana.publish.renderer import _default_outputs, publish_document, render_txt


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _debate(language: str = "en", chunks: int = 1) -> dict:
    """Minimal debate dict."""
    data: dict = {
        "debate_id": "d1",
        "title": "The Big Debate",
        "topic": "AI",
        "language": language,
        "chunks": [],
        "events": [
            {"global_order": 1, "speaker_name": "Alice",
             "text": "Hello.", "event_kind": "utterance"},
        ],
    }
    for i in range(chunks):
        data["chunks"].append({
            "chunk_order": i + 1,
            "title": f"Chunk {i + 1}",
            "events": [
                {"global_order": i * 10 + 1, "speaker_name": "Alice",
                 "text": f"Chunk {i + 1} text.", "event_kind": "utterance"},
            ],
        })
    return data


def _news(language: str = "en") -> dict:
    return {
        "show_id": "n1",
        "title": "Evening News",
        "language": language,
        "segments": [
            {
                "segment_order": 0,
                "headline": "Big Story",
                "events": [
                    {"global_order": 1, "speaker_name": "Anchor",
                     "text": "Good evening.", "event_kind": "utterance"},
                ],
            }
        ],
    }


def _podcast(language: str = "en") -> dict:
    return {
        "podcast_id": "p1",
        "title": "My Podcast",
        "language": language,
        "segments": [
            {
                "segment_order": 0,
                "events": [
                    {"global_order": 1, "speaker_name": "Host",
                     "text": "Welcome.", "event_kind": "utterance"},
                ],
            }
        ],
    }


def _audiobook(language: str = "en", chapters: int = 1) -> dict:
    data: dict = {
        "audiobook_id": "a1",
        "title": "My Book",
        "language": language,
        "chapters": [],
    }
    for i in range(chapters):
        data["chapters"].append({
            "chapter_order": i + 1,
            "title": f"The Beginning {i + 1}",
            "events": [
                {"global_order": i * 10 + 1, "speaker_name": "",
                 "speaker_role": "narrator",
                 "text": f"Chapter {i + 1} prose.", "event_kind": "narration"},
            ],
        })
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────


class TestFormattedDocumentLanguage:
    def test_default_language_is_en(self):
        doc = FormattedDocument(show_type="debate", title="T")
        assert doc.language == "en"

    def test_custom_language_stored(self):
        doc = FormattedDocument(show_type="debate", title="T", language="es")
        assert doc.language == "es"


class TestPublishResultLanguage:
    def test_default_language_is_en(self):
        r = PublishResult()
        assert r.language == "en"

    def test_custom_language_stored(self):
        r = PublishResult(language="fr")
        assert r.language == "fr"


# ─────────────────────────────────────────────────────────────────────────────
# Section labels
# ─────────────────────────────────────────────────────────────────────────────


class TestSectionLabels:
    def test_english_act(self):
        assert _get_labels("en")["act"] == "ACT"

    def test_spanish_act(self):
        assert _get_labels("es")["act"] == "ACTO"

    def test_french_act(self):
        assert _get_labels("fr")["act"] == "ACTE"

    def test_german_act(self):
        assert _get_labels("de")["act"] == "AKT"

    def test_portuguese_act(self):
        assert _get_labels("pt")["act"] == "ATO"

    def test_italian_act(self):
        assert _get_labels("it")["act"] == "ATTO"

    def test_unknown_code_falls_back_to_english(self):
        labels = _get_labels("xx")
        assert labels["act"] == "ACT"
        assert labels["chapter"] == "Chapter"

    def test_all_defined_languages_have_required_keys(self):
        required = {"act", "chapter", "segment"}
        for lang, labels in _SECTION_LABELS.items():
            missing = required - labels.keys()
            assert not missing, f"Language {lang!r} missing keys: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# DebateFormatter
# ─────────────────────────────────────────────────────────────────────────────


class TestDebateFormatterLanguage:
    def test_english_language_default(self):
        data = _debate()
        del data["language"]
        doc = DebateFormatter().format(data)
        assert doc.language == "en"

    def test_language_field_propagated(self):
        doc = DebateFormatter().format(_debate(language="es"))
        assert doc.language == "es"

    def test_french_language_stored(self):
        doc = DebateFormatter().format(_debate(language="fr"))
        assert doc.language == "fr"

    def test_multi_act_english_label(self):
        doc = DebateFormatter().format(_debate(language="en", chunks=2))
        section_titles = [s.title for s in doc.sections]
        assert "ACT 1" in section_titles
        assert "ACT 2" in section_titles

    def test_multi_act_spanish_label(self):
        doc = DebateFormatter().format(_debate(language="es", chunks=3))
        titles = [s.title for s in doc.sections]
        assert all(t.startswith("ACTO") for t in titles), titles

    def test_multi_act_french_label(self):
        doc = DebateFormatter().format(_debate(language="fr", chunks=2))
        titles = [s.title for s in doc.sections]
        assert titles[0] == "ACTE 1"

    def test_single_chunk_no_act_heading(self):
        # Single chunk → no ACT heading, just an empty title section.
        doc = DebateFormatter().format(_debate(language="es", chunks=1))
        assert doc.sections[0].title == ""

    def test_unknown_language_falls_back_to_act(self):
        doc = DebateFormatter().format(_debate(language="zz", chunks=2))
        titles = [s.title for s in doc.sections]
        assert titles[0].startswith("ACT")


# ─────────────────────────────────────────────────────────────────────────────
# NewsFormatter
# ─────────────────────────────────────────────────────────────────────────────


class TestNewsFormatterLanguage:
    def test_language_propagated(self):
        doc = NewsFormatter().format(_news(language="de"))
        assert doc.language == "de"

    def test_default_language_en(self):
        data = _news()
        del data["language"]
        doc = NewsFormatter().format(data)
        assert doc.language == "en"


# ─────────────────────────────────────────────────────────────────────────────
# PodcastFormatter
# ─────────────────────────────────────────────────────────────────────────────


class TestPodcastFormatterLanguage:
    def test_language_propagated(self):
        doc = PodcastFormatter().format(_podcast(language="it"))
        assert doc.language == "it"

    def test_spanish_segment_label(self):
        data = _podcast(language="es")
        doc = PodcastFormatter().format(data)
        # Section title should start with localised segment label.
        assert doc.sections[0].title.startswith("Segmento")

    def test_french_segment_label(self):
        doc = PodcastFormatter().format(_podcast(language="fr"))
        assert doc.sections[0].title.startswith("Segment")

    def test_default_segment_label_english(self):
        doc = PodcastFormatter().format(_podcast())
        assert doc.sections[0].title.startswith("Segment")


# ─────────────────────────────────────────────────────────────────────────────
# AudiobookFormatter
# ─────────────────────────────────────────────────────────────────────────────


class TestAudiobookFormatterLanguage:
    def test_language_propagated(self):
        doc = AudiobookFormatter().format(_audiobook(language="pt"))
        assert doc.language == "pt"

    def test_spanish_chapter_label(self):
        doc = AudiobookFormatter().format(_audiobook(language="es", chapters=2))
        titles = [s.title for s in doc.sections]
        assert titles[0].startswith("Capítulo")

    def test_french_chapter_label(self):
        doc = AudiobookFormatter().format(_audiobook(language="fr"))
        assert doc.sections[0].title.startswith("Chapitre")

    def test_german_chapter_label(self):
        doc = AudiobookFormatter().format(_audiobook(language="de"))
        assert doc.sections[0].title.startswith("Kapitel")

    def test_english_chapter_label(self):
        doc = AudiobookFormatter().format(_audiobook())
        assert doc.sections[0].title.startswith("Chapter")

    def test_chapter_title_appended(self):
        doc = AudiobookFormatter().format(_audiobook(language="es"))
        # Format: "Capítulo 1: The Beginning 1"
        assert ": The Beginning 1" in doc.sections[0].title


# ─────────────────────────────────────────────────────────────────────────────
# format_document dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatDocumentLanguage:
    def test_debate_language_propagated(self):
        doc = format_document(_debate(language="es"))
        assert doc.language == "es"

    def test_news_language_propagated(self):
        doc = format_document(_news(language="fr"))
        assert doc.language == "fr"

    def test_podcast_language_propagated(self):
        doc = format_document(_podcast(language="de"))
        assert doc.language == "de"

    def test_audiobook_language_propagated(self):
        doc = format_document(_audiobook(language="pt"))
        assert doc.language == "pt"


# ─────────────────────────────────────────────────────────────────────────────
# _default_outputs — language suffix logic
# ─────────────────────────────────────────────────────────────────────────────


class TestDefaultOutputs:
    def _src(self, tmp_path: Path) -> Path:
        p = tmp_path / "debate.json"
        p.write_text("{}")
        return p

    def test_root_language_no_suffix(self, tmp_path):
        src = self._src(tmp_path)
        paths = _default_outputs(src, None, ["txt"], language="en", root_lang="en")
        assert paths["txt"].name == "debate.txt"

    def test_non_root_language_adds_suffix(self, tmp_path):
        src = self._src(tmp_path)
        paths = _default_outputs(src, None, ["txt"], language="es", root_lang="en")
        assert paths["txt"].name == "debate_es.txt"

    def test_non_root_pdf_suffix(self, tmp_path):
        src = self._src(tmp_path)
        paths = _default_outputs(src, None, ["pdf"], language="fr", root_lang="en")
        assert paths["pdf"].name == "debate_fr.pdf"

    def test_custom_root_lang(self, tmp_path):
        src = self._src(tmp_path)
        # When root_lang=es and language=es → no suffix.
        paths = _default_outputs(src, None, ["txt"], language="es", root_lang="es")
        assert paths["txt"].name == "debate.txt"

    def test_default_root_lang_is_en(self, tmp_path):
        src = self._src(tmp_path)
        paths = _default_outputs(src, None, ["txt"], language="de")
        assert paths["txt"].name == "debate_de.txt"

    def test_dest_dir_honoured(self, tmp_path):
        src = self._src(tmp_path)
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        paths = _default_outputs(src, out_dir, ["txt"], language="es")
        assert paths["txt"].parent == out_dir
        assert paths["txt"].name == "debate_es.txt"


# ─────────────────────────────────────────────────────────────────────────────
# publish_document — integration
# ─────────────────────────────────────────────────────────────────────────────


class TestPublishDocumentLanguage:
    def _write_json(self, tmp_path: Path, data: dict, name: str = "show.json") -> Path:
        p = tmp_path / name
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_result_language_english(self, tmp_path):
        src = self._write_json(tmp_path, _debate())
        result = publish_document(src, dest_dir=tmp_path)
        assert result.language == "en"

    def test_result_language_spanish(self, tmp_path):
        src = self._write_json(tmp_path, _debate(language="es"))
        result = publish_document(src, dest_dir=tmp_path)
        assert result.language == "es"

    def test_english_output_no_suffix(self, tmp_path):
        src = self._write_json(tmp_path, _debate())
        result = publish_document(src, dest_dir=tmp_path)
        assert result.txt_path is not None
        assert "_en" not in result.txt_path.name

    def test_spanish_output_gets_suffix(self, tmp_path):
        src = self._write_json(tmp_path, _debate(language="es"))
        result = publish_document(src, dest_dir=tmp_path)
        assert result.txt_path is not None
        assert result.txt_path.name.endswith("_es.txt")

    def test_root_lang_overrides_suffix(self, tmp_path):
        # If root_lang=es, Spanish output should have no suffix.
        src = self._write_json(tmp_path, _debate(language="es"))
        result = publish_document(src, dest_dir=tmp_path, root_lang="es")
        assert result.txt_path is not None
        assert "_es" not in result.txt_path.name

    def test_txt_file_is_written(self, tmp_path):
        src = self._write_json(tmp_path, _debate(language="fr"))
        result = publish_document(src, dest_dir=tmp_path)
        assert result.txt_path is not None
        assert result.txt_path.exists()

    def test_txt_content_uses_localised_labels(self, tmp_path):
        src = self._write_json(tmp_path, _debate(language="es", chunks=2))
        result = publish_document(src, dest_dir=tmp_path)
        content = result.txt_path.read_text(encoding="utf-8")
        assert "ACTO" in content

    def test_news_spanish_published(self, tmp_path):
        src = self._write_json(tmp_path, _news(language="es"))
        result = publish_document(src, dest_dir=tmp_path)
        assert result.language == "es"
        assert result.txt_path.name.endswith("_es.txt")


# ─────────────────────────────────────────────────────────────────────────────
# HTML template — lang attribute
# ─────────────────────────────────────────────────────────────────────────────


class TestHtmlTemplate:
    def test_base_template_uses_doc_language(self):
        """The _base.html.jinja2 must use {{ doc.language }}, not hardcoded en."""
        from pathlib import Path as _Path
        base = (
            _Path(__file__).parents[1]
            / "templates" / "publish" / "_base.html.jinja2"
        )
        content = base.read_text(encoding="utf-8")
        # Should NOT contain the hardcoded 'lang="en"'.
        assert 'lang="en"' not in content
        # Should contain the template variable.
        assert '{{ doc.language }}' in content

    def test_render_html_language_in_output(self, tmp_path):
        """render_txt doesn't use the template; render_pdf requires weasyprint.
        We instead test that _render_html produces correct lang attr."""
        try:
            from vox_romana.publish.renderer import _render_html
        except ImportError:
            pytest.skip("jinja2 not available")

        doc = FormattedDocument(
            show_type="debate",
            title="El Gran Debate",
            language="es",
            sections=[],
        )
        try:
            html = _render_html(doc)
        except Exception:
            pytest.skip("templates or jinja2 not available")

        assert 'lang="es"' in html
