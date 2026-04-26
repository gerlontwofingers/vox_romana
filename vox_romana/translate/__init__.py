"""vox_romana.translate — pluggable translation engine.

Provides a Translator ABC and a concrete LLMTranslator implementation,
plus the translate_show_data() function that translates an entire show
JSON dictionary from one language to another.

Usage::

    from vox_romana.translate import LLMTranslator, translate_show_data
    from vox_romana.core.client import DeepSeekClient

    client = DeepSeekClient()
    translator = LLMTranslator(client)
    translated = translate_show_data(show_data, target_lang="es", translator=translator)

Adding a new translation backend:
    Subclass Translator and implement translate().  Override
    translate_batch() when the backend has a native batch/bulk API
    (DeepL, Google Translate, etc.) for better throughput.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


# ── Public exception ─────────────────────────────────────────────────────────


class TranslationError(RuntimeError):
    """Raised when a translation call fails or returns unexpected output."""


# ── Abstract base class ───────────────────────────────────────────────────────


class Translator(ABC):
    """Abstract base class for all translation backends.

    Implement :meth:`translate` at minimum.  Override
    :meth:`translate_batch` when the backend supports multi-item calls
    (LLM JSON arrays, DeepL, Google Translate, etc.) for efficiency.
    """

    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single string.

        Args:
            text:        The source text to translate.
            source_lang: BCP-47 code of the source language (e.g. ``"en"``).
            target_lang: BCP-47 code of the target language (e.g. ``"es"``).

        Returns:
            The translated string.

        Raises:
            TranslationError: If translation fails.
        """

    def translate_batch(
        self,
        texts: Sequence[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate a sequence of strings.

        Default implementation calls :meth:`translate` for each item.
        Override in subclasses that support bulk API calls.

        Args:
            texts:       Sequence of source strings.
            source_lang: BCP-47 code of the source language.
            target_lang: BCP-47 code of the target language.

        Returns:
            List of translated strings, same length and order as ``texts``.

        Raises:
            TranslationError: If any translation fails.
        """
        return [self.translate(t, source_lang, target_lang) for t in texts]


# ── Concrete implementations (imported after ABC to avoid circular imports) ──

from .llm_translator import LLMTranslator  # noqa: E402
from .event_translator import translate_show_data  # noqa: E402

__all__ = [
    "Translator",
    "TranslationError",
    "LLMTranslator",
    "translate_show_data",
]
