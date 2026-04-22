"""Tests for vox_romana/translate/ — Shot 2."""

from __future__ import annotations

import json
from typing import Sequence
from unittest.mock import MagicMock, call, patch

import pytest

from vox_romana.translate import (
    LLMTranslator,
    Translator,
    TranslationError,
    translate_show_data,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────


class _UpperTranslator(Translator):
    """Toy translator: uppercases every string, no LLM required.

    Used in translate_show_data tests so we can assert field-by-field
    what was translated (translated text == original.upper()) without
    any mocking.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        self.calls.append((text, source_lang, target_lang))
        return text.upper()

    def translate_batch(
        self,
        texts: Sequence[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        return [self.translate(t, source_lang, target_lang) for t in texts]


class _NullTranslator(Translator):
    """Translator that records whether it was called, but does nothing."""

    def __init__(self) -> None:
        self.called = False

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        self.called = True
        return text


def _mock_client(responses: list[list[str]]) -> MagicMock:
    """Return a mock DeepSeekClient whose .chat() yields successive JSON arrays."""
    client = MagicMock()
    client.model = "deepseek-chat"
    client.api_key = "sk-test"
    client.chat.side_effect = [json.dumps(r, ensure_ascii=False) for r in responses]
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Translator ABC
# ─────────────────────────────────────────────────────────────────────────────


class TestTranslatorABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Translator()  # type: ignore[abstract]

    def test_default_translate_batch_calls_translate(self):
        class _Concrete(Translator):
            calls: list[str] = []

            def translate(self, text: str, src: str, tgt: str) -> str:
                self.calls.append(text)
                return text + "_t"

        t = _Concrete()
        result = t.translate_batch(["a", "b", "c"], "en", "es")
        assert result == ["a_t", "b_t", "c_t"]
        assert t.calls == ["a", "b", "c"]

    def test_translate_batch_preserves_order(self):
        class _Rev(Translator):
            def translate(self, text: str, src: str, tgt: str) -> str:
                return text[::-1]

        t = _Rev()
        result = t.translate_batch(["abc", "def"], "en", "fr")
        assert result == ["cba", "fed"]


# ─────────────────────────────────────────────────────────────────────────────
# TranslationError
# ─────────────────────────────────────────────────────────────────────────────


class TestTranslationError:
    def test_is_runtime_error(self):
        assert issubclass(TranslationError, RuntimeError)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(TranslationError, match="oops"):
            raise TranslationError("oops")


# ─────────────────────────────────────────────────────────────────────────────
# LLMTranslator — basic
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMTranslatorBasic:
    def test_translate_single(self):
        client = _mock_client([["Hola mundo."]])
        t = LLMTranslator(client)
        assert t.translate("Hello world.", "en", "es") == "Hola mundo."

    def test_translate_batch_empty_returns_empty(self):
        client = MagicMock()
        t = LLMTranslator(client)
        result = t.translate_batch([], "en", "es")
        assert result == []
        client.chat.assert_not_called()

    def test_translate_batch_multiple(self):
        client = _mock_client([["Uno.", "Dos.", "Tres."]])
        t = LLMTranslator(client)
        result = t.translate_batch(["One.", "Two.", "Three."], "en", "es")
        assert result == ["Uno.", "Dos.", "Tres."]

    def test_translate_batch_preserves_order(self):
        client = _mock_client([["A", "B", "C"]])
        t = LLMTranslator(client)
        result = t.translate_batch(["x", "y", "z"], "en", "fr")
        assert result == ["A", "B", "C"]

    def test_is_translator_subclass(self):
        client = MagicMock()
        t = LLMTranslator(client)
        assert isinstance(t, Translator)


# ─────────────────────────────────────────────────────────────────────────────
# LLMTranslator — batching
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMTranslatorBatching:
    def test_small_batch_single_call(self):
        client = _mock_client([["A", "B"]])
        t = LLMTranslator(client)
        t.translate_batch(["a", "b"], "en", "es")
        assert client.chat.call_count == 1

    def test_batch_size_splits_correctly(self):
        # BATCH_SIZE=2, 5 items → 3 LLM calls (2+2+1)
        client = _mock_client([["A", "B"], ["C", "D"], ["E"]])
        t = LLMTranslator(client)
        t.BATCH_SIZE = 2
        result = t.translate_batch(["a", "b", "c", "d", "e"], "en", "es")
        assert client.chat.call_count == 3
        assert result == ["A", "B", "C", "D", "E"]

    def test_exact_batch_size_single_call(self):
        client = _mock_client([["T1", "T2"]])
        t = LLMTranslator(client)
        t.BATCH_SIZE = 2
        t.translate_batch(["a", "b"], "en", "fr")
        assert client.chat.call_count == 1

    def test_batch_size_plus_one_two_calls(self):
        client = _mock_client([["T1", "T2"], ["T3"]])
        t = LLMTranslator(client)
        t.BATCH_SIZE = 2
        t.translate_batch(["a", "b", "c"], "en", "de")
        assert client.chat.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# LLMTranslator — response parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMTranslatorParsing:
    def _translator_with_response(self, raw_response: str) -> LLMTranslator:
        client = MagicMock()
        client.model = "deepseek-chat"
        client.api_key = "sk-test"
        client.chat.return_value = raw_response
        return LLMTranslator(client)

    def test_plain_json_array(self):
        t = self._translator_with_response('["Hola"]')
        assert t.translate("Hello", "en", "es") == "Hola"

    def test_markdown_fence_stripped(self):
        t = self._translator_with_response('```json\n["Hola"]\n```')
        assert t.translate("Hello", "en", "es") == "Hola"

    def test_plain_fence_stripped(self):
        t = self._translator_with_response('```\n["Hola"]\n```')
        assert t.translate("Hello", "en", "es") == "Hola"

    def test_fallback_regex_extraction(self):
        # LLM wraps its response in extra prose
        t = self._translator_with_response('Here is the translation: ["Hola"] — done.')
        assert t.translate("Hello", "en", "es") == "Hola"

    def test_malformed_response_raises(self):
        t = self._translator_with_response("Sorry, I cannot translate that.")
        with pytest.raises(TranslationError):
            t.translate("Hello", "en", "es")

    def test_wrong_count_raises(self):
        # Send 2 items but LLM returns 1
        client = _mock_client([["Uno"]])
        t = LLMTranslator(client)
        with pytest.raises(TranslationError, match="2"):
            t.translate_batch(["One", "Two"], "en", "es")

    def test_non_list_json_raises(self):
        t = self._translator_with_response('{"key": "value"}')
        with pytest.raises(TranslationError):
            t.translate("Hello", "en", "es")

    def test_non_string_items_coerced(self):
        # LLM returns numbers — should be coerced to strings
        t = self._translator_with_response("[42]")
        result = t.translate("Hello", "en", "es")
        assert result == "42"

    def test_unicode_preserved(self):
        client = _mock_client([["こんにちは"]])
        t = LLMTranslator(client)
        assert t.translate("Hello", "en", "ja") == "こんにちは"


# ─────────────────────────────────────────────────────────────────────────────
# LLMTranslator — model override
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMTranslatorModelOverride:
    def test_same_model_reuses_client(self):
        client = MagicMock()
        client.model = "deepseek-chat"
        client.api_key = "sk-test"
        client.chat.return_value = '["ok"]'
        t = LLMTranslator(client, model="deepseek-chat")
        assert t._client is client

    def test_no_model_reuses_client(self):
        client = MagicMock()
        client.model = "deepseek-chat"
        t = LLMTranslator(client)
        assert t._client is client

    def test_different_model_creates_new_client(self):
        original_client = MagicMock()
        original_client.model = "deepseek-chat"
        original_client.api_key = "sk-test"

        with patch("vox_romana.core.client.DeepSeekClient") as MockClient:
            new_client = MagicMock()
            new_client.chat.return_value = '["ok"]'
            MockClient.return_value = new_client

            t = LLMTranslator(original_client, model="deepseek-reasoner")
            MockClient.assert_called_once_with(
                api_key="sk-test", model="deepseek-reasoner"
            )
            assert t._client is new_client


# ─────────────────────────────────────────────────────────────────────────────
# LLMTranslator — system prompt content
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMTranslatorSystemPrompt:
    def test_system_prompt_contains_language_names(self):
        client = _mock_client([["ok"]])
        t = LLMTranslator(client)
        t.translate("hello", "en", "es")
        system_msg = client.chat.call_args[0][0][0]
        assert system_msg["role"] == "system"
        content = system_msg["content"]
        assert "English" in content
        assert "Spanish" in content

    def test_user_message_is_json_array(self):
        client = _mock_client([["ok"]])
        t = LLMTranslator(client)
        t.translate("hello world", "en", "fr")
        user_msg = client.chat.call_args[0][0][1]
        assert user_msg["role"] == "user"
        parsed = json.loads(user_msg["content"])
        assert parsed == ["hello world"]


# ─────────────────────────────────────────────────────────────────────────────
# translate_show_data — basic behaviour
# ─────────────────────────────────────────────────────────────────────────────


class TestTranslateShowDataBasic:
    def _debate(self, **kwargs) -> dict:
        base = {
            "debate_id": "d1",
            "title": "The Great Debate",
            "topic": "Universal Basic Income",
            "setting": "A university auditorium",
            "language": "en",
            "chunks": [],
            "events": [],
        }
        base.update(kwargs)
        return base

    def test_translates_title(self):
        t = _UpperTranslator()
        result = translate_show_data(self._debate(), "es", t)
        assert result["title"] == "THE GREAT DEBATE"

    def test_translates_topic(self):
        t = _UpperTranslator()
        result = translate_show_data(self._debate(), "es", t)
        assert result["topic"] == "UNIVERSAL BASIC INCOME"

    def test_translates_setting(self):
        t = _UpperTranslator()
        result = translate_show_data(self._debate(), "es", t)
        assert result["setting"] == "A UNIVERSITY AUDITORIUM"

    def test_sets_language_field(self):
        t = _UpperTranslator()
        result = translate_show_data(self._debate(), "fr", t)
        assert result["language"] == "fr"

    def test_does_not_mutate_input(self):
        data = self._debate()
        original_title = data["title"]
        t = _UpperTranslator()
        translate_show_data(data, "es", t)
        assert data["title"] == original_title

    def test_returns_new_dict(self):
        data = self._debate()
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result is not data

    def test_same_lang_no_translation_called(self):
        data = self._debate(language="es")
        t = _NullTranslator()
        translate_show_data(data, "es", t)
        assert not t.called

    def test_same_lang_stamps_language(self):
        data = self._debate(language="es")
        data["language"] = "es"
        t = _NullTranslator()
        result = translate_show_data(data, "es", t)
        assert result["language"] == "es"

    def test_source_lang_inferred_from_data(self):
        data = self._debate(language="fr")
        t = _UpperTranslator()
        translate_show_data(data, "de", t)
        # All calls should have source_lang="fr"
        for _, src, _ in t.calls:
            assert src == "fr"

    def test_source_lang_override(self):
        data = self._debate(language="en")
        t = _UpperTranslator()
        translate_show_data(data, "es", t, source_lang="pt")
        for _, src, _ in t.calls:
            assert src == "pt"

    def test_target_lang_propagated(self):
        data = self._debate()
        t = _UpperTranslator()
        translate_show_data(data, "ja", t)
        for _, _, tgt in t.calls:
            assert tgt == "ja"

    def test_missing_language_field_defaults_to_en(self):
        data = self._debate()
        del data["language"]
        t = _UpperTranslator()
        translate_show_data(data, "es", t)
        for _, src, _ in t.calls:
            assert src == "en"


# ─────────────────────────────────────────────────────────────────────────────
# translate_show_data — events
# ─────────────────────────────────────────────────────────────────────────────


class TestTranslateShowDataEvents:
    def _event(self, text: str, kind: str = "utterance") -> dict:
        return {
            "global_order": 1,
            "speaker_name": "Alice",
            "text": text,
            "event_kind": kind,
        }

    def test_translates_utterance_text(self):
        data = {
            "debate_id": "d1",
            "title": "T",
            "language": "en",
            "chunks": [],
            "events": [self._event("Hello there.")],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["events"][0]["text"] == "HELLO THERE."

    def test_translates_narration_event(self):
        data = {
            "show_id": "s1",
            "title": "T",
            "language": "en",
            "events": [self._event("The scene opens.", kind="narration")],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["events"][0]["text"] == "THE SCENE OPENS."

    def test_translates_interrupt_event(self):
        data = {
            "debate_id": "d1",
            "title": "T",
            "language": "en",
            "chunks": [],
            "events": [self._event("Wait—", kind="interrupt")],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["events"][0]["text"] == "WAIT—"

    def test_skips_non_spoken_event_kinds(self):
        data = {
            "show_id": "s1",
            "title": "T",
            "language": "en",
            "events": [self._event("♪ theme music ♪", kind="bumper")],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        # "bumper" is not a spoken kind — text should be unchanged
        assert result["events"][0]["text"] == "♪ theme music ♪"

    def test_skips_none_text(self):
        event = {
            "global_order": 1,
            "speaker_name": "Alice",
            "text": None,
            "event_kind": "utterance",
        }
        data = {"debate_id": "d1", "title": "T", "language": "en", "chunks": [], "events": [event]}
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["events"][0]["text"] is None

    def test_skips_empty_text(self):
        data = {
            "debate_id": "d1",
            "title": "T",
            "language": "en",
            "chunks": [],
            "events": [self._event("")],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["events"][0]["text"] == ""

    def test_skips_whitespace_only_text(self):
        data = {
            "debate_id": "d1",
            "title": "T",
            "language": "en",
            "chunks": [],
            "events": [self._event("   ")],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["events"][0]["text"] == "   "

    def test_multiple_events_all_translated(self):
        events = [
            self._event("First utterance."),
            self._event("Second utterance."),
            self._event("Third utterance."),
        ]
        data = {"debate_id": "d1", "title": "T", "language": "en", "chunks": [], "events": events}
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        for ev in result["events"]:
            assert ev["text"] == ev["text"].upper()


# ─────────────────────────────────────────────────────────────────────────────
# translate_show_data — debate chunks
# ─────────────────────────────────────────────────────────────────────────────


class TestTranslateShowDataDebateChunks:
    def _chunk(self, title: str, events: list | None = None) -> dict:
        return {
            "chunk_order": 1,
            "title": title,
            "topic": "The topic",
            "setting": "The setting",
            "events": events or [],
        }

    def test_translates_chunk_title(self):
        data = {
            "debate_id": "d1",
            "title": "Root",
            "language": "en",
            "chunks": [self._chunk("Act One")],
            "events": [],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["chunks"][0]["title"] == "ACT ONE"

    def test_translates_chunk_topic(self):
        data = {
            "debate_id": "d1",
            "title": "Root",
            "language": "en",
            "chunks": [self._chunk("Act One")],
            "events": [],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["chunks"][0]["topic"] == "THE TOPIC"

    def test_translates_chunk_setting(self):
        data = {
            "debate_id": "d1",
            "title": "Root",
            "language": "en",
            "chunks": [self._chunk("Act One")],
            "events": [],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["chunks"][0]["setting"] == "THE SETTING"

    def test_translates_events_inside_chunks(self):
        events = [
            {"global_order": 1, "speaker_name": "A", "text": "Inside chunk.", "event_kind": "utterance"},
        ]
        data = {
            "debate_id": "d1",
            "title": "T",
            "language": "en",
            "chunks": [self._chunk("Act", events=events)],
            "events": [],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["chunks"][0]["events"][0]["text"] == "INSIDE CHUNK."

    def test_multiple_chunks_all_translated(self):
        data = {
            "debate_id": "d1",
            "title": "T",
            "language": "en",
            "chunks": [self._chunk("Act One"), self._chunk("Act Two")],
            "events": [],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["chunks"][0]["title"] == "ACT ONE"
        assert result["chunks"][1]["title"] == "ACT TWO"


# ─────────────────────────────────────────────────────────────────────────────
# translate_show_data — news segments
# ─────────────────────────────────────────────────────────────────────────────


class TestTranslateShowDataNewsSegments:
    def _segment(self, headline: str, events: list | None = None) -> dict:
        return {
            "segment_id": "seg1",
            "segment_order": 0,
            "headline": headline,
            "events": events or [],
        }

    def test_translates_segment_headline(self):
        data = {
            "show_id": "s1",
            "title": "Evening Brief",
            "language": "en",
            "segments": [self._segment("Breaking: Storm hits coast")],
            "events": [],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["segments"][0]["headline"] == "BREAKING: STORM HITS COAST"

    def test_translates_segment_events(self):
        events = [
            {"global_order": 1, "speaker_name": "Anchor", "text": "Good evening.", "event_kind": "utterance"},
        ]
        data = {
            "show_id": "s1",
            "title": "T",
            "language": "en",
            "segments": [self._segment("Headline", events=events)],
            "events": [],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["segments"][0]["events"][0]["text"] == "GOOD EVENING."

    def test_multiple_segments(self):
        data = {
            "show_id": "s1",
            "title": "T",
            "language": "en",
            "segments": [
                self._segment("Story One"),
                self._segment("Story Two"),
            ],
            "events": [],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["segments"][0]["headline"] == "STORY ONE"
        assert result["segments"][1]["headline"] == "STORY TWO"

    def test_missing_headline_key_ignored(self):
        segment = {"segment_id": "seg1", "segment_order": 0, "events": []}
        data = {
            "show_id": "s1",
            "title": "T",
            "language": "en",
            "segments": [segment],
            "events": [],
        }
        t = _UpperTranslator()
        # Should not crash
        result = translate_show_data(data, "es", t)
        assert "headline" not in result["segments"][0]


# ─────────────────────────────────────────────────────────────────────────────
# translate_show_data — non-translatable fields preserved
# ─────────────────────────────────────────────────────────────────────────────


class TestTranslateShowDataPreservation:
    def test_debate_id_unchanged(self):
        data = {
            "debate_id": "abc-123",
            "title": "T",
            "language": "en",
            "chunks": [],
            "events": [],
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["debate_id"] == "abc-123"

    def test_speaker_name_unchanged(self):
        event = {"global_order": 1, "speaker_name": "Dr. Chen", "text": "Hello.", "event_kind": "utterance"}
        data = {"debate_id": "d1", "title": "T", "language": "en", "chunks": [], "events": [event]}
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["events"][0]["speaker_name"] == "Dr. Chen"

    def test_voice_casting_unchanged(self):
        data = {
            "debate_id": "d1",
            "title": "T",
            "language": "en",
            "chunks": [],
            "events": [],
            "voice_casting": {"Alice": "female", "Bob": "male"},
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["voice_casting"] == {"Alice": "female", "Bob": "male"}

    def test_event_kind_unchanged(self):
        event = {"global_order": 1, "speaker_name": "A", "text": "Hi.", "event_kind": "utterance"}
        data = {"debate_id": "d1", "title": "T", "language": "en", "chunks": [], "events": [event]}
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["events"][0]["event_kind"] == "utterance"

    def test_unknown_top_level_keys_preserved(self):
        data = {
            "debate_id": "d1",
            "title": "T",
            "language": "en",
            "chunks": [],
            "events": [],
            "custom_field": "keep me",
        }
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["custom_field"] == "keep me"

    def test_empty_show_no_crash(self):
        data = {"show_id": "s1", "title": "T", "language": "en"}
        t = _UpperTranslator()
        result = translate_show_data(data, "es", t)
        assert result["language"] == "es"
