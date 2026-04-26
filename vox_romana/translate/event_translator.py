"""Translate an entire show-data dict from one language to another.

:func:`translate_show_data` is the high-level entry point: it deep-copies
the input dict, collects every human-readable text field, batch-translates
them in as few LLM calls as possible, writes the translations back in
place, and stamps the output with the target language code.

Supported show types
--------------------
debate JSON
    Top-level:   title, topic, setting
    Per chunk:   title, topic, setting
    Per event:   text  (utterance / narration / interrupt kinds only)

news JSON
    Top-level:   title, topic
    Per segment: headline
    Per event:   text  (utterance / narration / interrupt kinds only)

Any unrecognised fields are left unchanged.  The function is deliberately
permissive — it will not crash on an unknown field, it just won't translate
it.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from . import Translator


# Event kinds whose ``text`` field contains spoken dialogue that should be
# translated.  Structural / audio-cue kinds (break, bumper, transition) are
# left untouched because they are internal pipeline tokens, not content.
_SPOKEN_EVENT_KINDS: frozenset[str] = frozenset({
    "utterance",
    "narration",
    "interrupt",
})

# String keys that hold human-readable content at the top level of a show
# dict and inside each debate chunk.
_COMMON_TEXT_KEYS: tuple[str, ...] = ("title", "topic", "setting")


def translate_show_data(
    data: dict[str, Any],
    target_lang: str,
    translator: Translator,
    *,
    source_lang: str | None = None,
) -> dict[str, Any]:
    """Deep-copy a show-data dict and translate all human-readable text.

    The input dict is never modified.  A fully independent deep copy is
    returned with translations applied and ``data["language"]`` updated.

    Args:
        data:        Show JSON as a Python dict (debate, news, etc.).
        target_lang: BCP-47 code of the target language (e.g. ``"es"``).
        translator:  A :class:`~vox_romana.translate.Translator` instance.
        source_lang: BCP-47 code of the source language.  Defaults to
                     ``data.get("language", "en")`` when not provided.

    Returns:
        A new dict with all translatable text fields replaced by their
        translations, and ``"language"`` set to ``target_lang``.

    Raises:
        TranslationError: Propagated from the translator if any batch fails.
    """
    result: dict[str, Any] = copy.deepcopy(data)
    src = source_lang or result.get("language") or "en"

    if src == target_lang:
        # Nothing to translate — stamp language and return.
        result["language"] = target_lang
        return result

    # ── Collect all translatable text fields ─────────────────────────────────
    # We gather (current_value, setter) pairs first, translate in one big
    # batch, then apply each setter.  This minimises LLM calls.

    texts: list[str] = []
    setters: list[Callable[[str], None]] = []

    def _collect(container: dict[str, Any], key: str) -> None:
        """Queue one string field for translation if it is non-empty."""
        val = container.get(key)
        if val and isinstance(val, str) and val.strip():
            texts.append(val)
            # Capture container + key by value to avoid late-binding gotcha.
            setters.append(
                lambda v, _c=container, _k=key: _c.__setitem__(_k, v)
            )

    def _collect_event(event: dict[str, Any]) -> None:
        """Queue an event's text when it represents spoken dialogue."""
        kind = event.get("event_kind", "utterance")
        if kind in _SPOKEN_EVENT_KINDS:
            _collect(event, "text")

    # Top-level common keys (present in both debate and news JSON)
    for key in _COMMON_TEXT_KEYS:
        _collect(result, key)

    # Debate: per-chunk keys + events inside each chunk
    for chunk in result.get("chunks", []):
        for key in _COMMON_TEXT_KEYS:
            _collect(chunk, key)
        for event in chunk.get("events", []):
            _collect_event(event)

    # Top-level events (present in both debate and news JSON)
    for event in result.get("events", []):
        _collect_event(event)

    # News: per-segment headline + events inside each segment
    for segment in result.get("segments", []):
        _collect(segment, "headline")
        for event in segment.get("events", []):
            _collect_event(event)

    # ── Batch-translate and apply ─────────────────────────────────────────────
    if texts:
        translated = translator.translate_batch(texts, src, target_lang)
        for setter, value in zip(setters, translated):
            setter(value)

    # ── Stamp the output language ─────────────────────────────────────────────
    result["language"] = target_lang

    return result
