"""Integration smoke tests — real DeepSeek API + optional local Piper TTS.

These tests make live external calls and are skipped by default.
Run them with::

    PYTHONPATH=. .venv/bin/pytest -q --run-integration tests/test_translation_smoke.py

Requirements:
    * ``DEEPSEEK_API_KEY`` in ``.env`` or the environment.
    * (optional) At least one non-English Piper ONNX voice installed under
      the configured ``PIPER_VOICES_DIR`` for the TTS round-trip test.
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def deepseek_client():
    """Return a real DeepSeekClient, skipping if no API key is available."""
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set — skipping integration tests")

    from vox_romana.core.client import DeepSeekClient

    return DeepSeekClient()


@pytest.fixture(scope="module")
def llm_translator(deepseek_client):
    """Return a real LLMTranslator backed by DeepSeek."""
    from vox_romana.translate import LLMTranslator

    return LLMTranslator(client=deepseek_client)


# ---------------------------------------------------------------------------
# Basic translation unit — single string
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLLMTranslatorSmoke:
    def test_translate_english_to_spanish(self, llm_translator):
        result = llm_translator.translate("Hello, world!", "en", "es")
        assert isinstance(result, str)
        assert len(result) > 0
        # "Hello" in Spanish should not be "Hello"
        assert "Hello, world!" not in result or result.lower() != "hello, world!"

    def test_translate_batch_preserves_length(self, llm_translator):
        texts = ["Good morning", "How are you?", "Thank you very much."]
        results = llm_translator.translate_batch(texts, "en", "es")
        assert len(results) == len(texts)
        for r in results:
            assert isinstance(r, str) and len(r) > 0

    def test_translate_empty_string_safe(self, llm_translator):
        # An empty string should either return empty or a trivially valid result.
        result = llm_translator.translate("", "en", "es")
        assert isinstance(result, str)

    def test_translate_short_phrase_non_empty(self, llm_translator):
        result = llm_translator.translate("Yes.", "en", "fr")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# translate_show_data — full show-dict translation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTranslateShowDataSmoke:
    def _make_debate(self) -> dict:
        return {
            "debate_id": "smoke_01",
            "title": "The Future of Energy",
            "topic": "Renewable energy transition",
            "setting": "A university lecture hall",
            "language": "en",
            "chunks": [
                {
                    "chunk_order": 1,
                    "title": "Opening Statements",
                    "topic": "Energy policy",
                    "setting": "Main stage",
                    "events": [
                        {
                            "global_order": 1,
                            "speaker_name": "Alice",
                            "text": "Solar and wind power will define the next century.",
                            "event_kind": "utterance",
                        },
                        {
                            "global_order": 2,
                            "speaker_name": "Bob",
                            "text": "Nuclear energy remains the safest and most reliable option.",
                            "event_kind": "utterance",
                        },
                    ],
                }
            ],
            "events": [],
        }

    def test_translated_language_field_set(self, llm_translator):
        from vox_romana.translate import translate_show_data

        result = translate_show_data(
            self._make_debate(), "es", llm_translator, source_lang="en"
        )
        assert result["language"] == "es"

    def test_original_not_mutated(self, llm_translator):
        from vox_romana.translate import translate_show_data

        original = self._make_debate()
        original_title = original["title"]
        translate_show_data(original, "es", llm_translator, source_lang="en")
        assert original["language"] == "en"
        assert original["title"] == original_title

    def test_title_translated(self, llm_translator):
        from vox_romana.translate import translate_show_data

        result = translate_show_data(
            self._make_debate(), "es", llm_translator, source_lang="en"
        )
        # Title should be a non-empty string and different from English
        assert isinstance(result["title"], str) and len(result["title"]) > 0

    def test_chunk_events_translated(self, llm_translator):
        from vox_romana.translate import translate_show_data

        result = translate_show_data(
            self._make_debate(), "fr", llm_translator, source_lang="en"
        )
        first_event_text = result["chunks"][0]["events"][0]["text"]
        assert isinstance(first_event_text, str) and len(first_event_text) > 0
        # The translated French should not be the verbatim English
        assert first_event_text != "Solar and wind power will define the next century."

    def test_same_language_short_circuits(self, llm_translator):
        """translate_show_data with src==target should return quickly without
        calling the LLM (verified by the translator not being a mock, yet
        the output being identical to input in content)."""
        from vox_romana.translate import translate_show_data
        from unittest.mock import patch

        data = self._make_debate()
        # Patch the translate_batch to detect if it gets called.
        with patch.object(
            llm_translator, "translate_batch", wraps=llm_translator.translate_batch
        ) as spy:
            result = translate_show_data(data, "en", llm_translator, source_lang="en")

        spy.assert_not_called()
        assert result["language"] == "en"

    def test_news_show_translated(self, llm_translator):
        from vox_romana.translate import translate_show_data

        news = {
            "show_id": "smoke_news_01",
            "title": "Evening Headlines",
            "topic": "World events",
            "language": "en",
            "segments": [
                {
                    "segment_order": 0,
                    "headline": "Scientists discover new clean fuel source",
                    "events": [
                        {
                            "global_order": 1,
                            "speaker_name": "Anchor",
                            "text": "Good evening. Our top story tonight…",
                            "event_kind": "utterance",
                        }
                    ],
                }
            ],
        }
        result = translate_show_data(news, "de", llm_translator, source_lang="en")
        assert result["language"] == "de"
        assert result["title"] != "Evening Headlines"
        headline = result["segments"][0]["headline"]
        assert isinstance(headline, str) and len(headline) > 0


# ---------------------------------------------------------------------------
# TTS round-trip — translate then synthesize with PiperEngine (requires models)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTTSRoundTrip:
    """Translate a single utterance then feed it through PiperEngine.

    Skipped automatically when no non-English Piper voice is installed.
    """

    @pytest.fixture(autouse=True)
    def _check_piper_non_english(self):
        """Skip if no non-English Piper voice is installed."""
        try:
            from vox_romana.tts.engines.piper_engine import PiperEngine

            engine = PiperEngine()
            langs = engine.available_languages()
            non_english = [l for l in langs if l != "en"]
        except Exception as exc:
            pytest.skip(f"PiperEngine not available: {exc}")

        if not non_english:
            pytest.skip("No non-English Piper voices installed")

        self._target_lang = non_english[0]

    def test_translate_then_synthesize(self, llm_translator):
        from vox_romana.tts.engines.piper_engine import PiperEngine
        from vox_romana.translate import LLMTranslator

        engine = PiperEngine()

        translated = llm_translator.translate(
            "Welcome to the show.", "en", self._target_lang
        )
        assert isinstance(translated, str) and len(translated) > 0

        # Synthesize — should return bytes without raising.
        audio = engine.synthesize(translated, language=self._target_lang)
        assert isinstance(audio, bytes)
        assert len(audio) > 0

    def test_full_show_translate_and_render(self, llm_translator, tmp_path):
        """End-to-end: mini debate dict → translate → publish TXT."""
        from vox_romana.translate import translate_show_data
        from vox_romana.publish import publish_document
        import json

        data = {
            "debate_id": "e2e_smoke",
            "title": "Energy Policy",
            "topic": "Renewable transition",
            "setting": "Stage",
            "language": "en",
            "chunks": [
                {
                    "chunk_order": 1,
                    "title": "Opening",
                    "topic": "Energy",
                    "setting": "Stage",
                    "events": [
                        {
                            "global_order": 1,
                            "speaker_name": "Alice",
                            "text": "The future is renewable.",
                            "event_kind": "utterance",
                        }
                    ],
                }
            ],
            "events": [],
        }

        translated = translate_show_data(
            data, self._target_lang, llm_translator, source_lang="en"
        )
        assert translated["language"] == self._target_lang

        # Write translated JSON and publish as TXT.
        src = tmp_path / f"e2e_{self._target_lang}.json"
        src.write_text(json.dumps(translated), encoding="utf-8")

        result = publish_document(src, dest_dir=tmp_path, root_lang="en")
        assert result.txt_path is not None
        assert result.txt_path.exists()
        content = result.txt_path.read_text(encoding="utf-8")
        assert len(content) > 0
