"""LLM-based translator using DeepSeek (or any compatible client).

Sends text to the configured LLM in JSON-array batches for efficiency.
Each batch call translates up to BATCH_SIZE strings in a single API call,
reducing cost and latency compared to one call per utterance.

The LLM is instructed to return a JSON array of the same length as the
input.  Markdown code fences are stripped automatically.  If the response
is malformed, a TranslationError is raised with the raw response included
for debugging.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from vox_romana.core.languages import LANGUAGE_NAMES

from . import Translator, TranslationError


def _lang_name(code: str) -> str:
    """Return the full display name for a BCP-47 code, or the code itself."""
    return LANGUAGE_NAMES.get(code, code)


class LLMTranslator(Translator):
    """Translate text via an LLM (DeepSeek or any client with a .chat() method).

    Batches strings as a JSON array and expects a JSON array response.
    Handles markdown stripping, length validation, and fallback regex parsing.

    Attributes:
        BATCH_SIZE: Maximum number of strings per LLM call.  Tune this
                    based on the model's context window and your latency
                    budget.  Default: 30.
    """

    BATCH_SIZE: int = 30

    def __init__(
        self,
        client: object,
        *,
        model: str | None = None,
    ) -> None:
        """Initialise the translator.

        Args:
            client: Any object with a ``.chat(messages) -> str`` method.
                    Typically a :class:`~vox_romana.core.client.DeepSeekClient`.
            model:  Optional model name override.  When provided and different
                    from the client's current model a new DeepSeekClient is
                    spawned with that model (the original client's API key is
                    reused).  Pass ``None`` to use the client's own default.
        """
        if model is not None and hasattr(client, "model") and model != client.model:
            from vox_romana.core.client import DeepSeekClient

            self._client = DeepSeekClient(api_key=client.api_key, model=model)
        else:
            self._client = client

    # ── Translator interface ──────────────────────────────────────────────────

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single string via the LLM."""
        return self.translate_batch([text], source_lang, target_lang)[0]

    def translate_batch(
        self,
        texts: Sequence[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate a sequence of strings, chunking into BATCH_SIZE LLM calls.

        Args:
            texts:       Strings to translate.
            source_lang: BCP-47 source language code.
            target_lang: BCP-47 target language code.

        Returns:
            Translated strings in the same order as input.

        Raises:
            TranslationError: If any LLM call returns malformed output.
        """
        text_list = list(texts)
        if not text_list:
            return []

        results: list[str] = []
        for i in range(0, len(text_list), self.BATCH_SIZE):
            batch = text_list[i : i + self.BATCH_SIZE]
            results.extend(self._call_llm(batch, source_lang, target_lang))
        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _system_prompt(self, source_lang: str, target_lang: str) -> str:
        src_name = _lang_name(source_lang)
        tgt_name = _lang_name(target_lang)
        return (
            f"You are a professional translator specialising in {src_name} to "
            f"{tgt_name} translation.\n"
            "The user will provide a JSON array of strings.  "
            "Return ONLY a JSON array of the same length with each string "
            f"translated into {tgt_name}.\n"
            "Rules:\n"
            "- Preserve proper nouns (people's names, place names, brand names)\n"
            "- Maintain the original tone and register "
            "(formal, conversational, technical, etc.)\n"
            "- Preserve punctuation style and paragraph breaks\n"
            "- Do NOT add explanations, notes, or extra commentary\n"
            "- Return ONLY the raw JSON array — no markdown code fences, "
            "no preamble, no trailing text"
        )

    def _call_llm(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Make one LLM call for a single batch and return translated strings."""
        user_payload = json.dumps(texts, ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(source_lang, target_lang),
            },
            {"role": "user", "content": user_payload},
        ]
        raw_response = self._client.chat(messages)
        return self._parse_response(raw_response, expected_count=len(texts))

    def _parse_response(self, response: str, *, expected_count: int) -> list[str]:
        """Parse the LLM response into a list of translated strings.

        Handles markdown code fences and falls back to regex extraction when
        direct JSON parsing fails.

        Args:
            response:       Raw string returned by the LLM.
            expected_count: Expected number of items (same as input batch size).

        Returns:
            List of translated strings.

        Raises:
            TranslationError: If the response cannot be parsed or has the
                              wrong number of items.
        """
        text = response.strip()

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Attempt 1: direct JSON parse
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract first JSON array found anywhere in the response
        if parsed is None:
            m = re.search(r"\[.*?\]", text, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            raise TranslationError(
                f"LLM returned an unparseable response "
                f"(first 300 chars): {response[:300]!r}"
            )

        if not isinstance(parsed, list):
            raise TranslationError(
                f"LLM returned {type(parsed).__name__} but a JSON array was expected."
            )

        if len(parsed) != expected_count:
            raise TranslationError(
                f"LLM returned {len(parsed)} item(s) but {expected_count} were sent. "
                f"Response (first 300 chars): {response[:300]!r}"
            )

        return [str(item) for item in parsed]
