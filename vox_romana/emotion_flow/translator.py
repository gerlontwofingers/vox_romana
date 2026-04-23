"""TranslatorAgent — maps speech segments to English using Helsinki-NLP opus-mt models.

Performs:
  1. Language detection via langdetect.
  2. Forward translation to English (if source != English).
  3. Back-translation to source language for quality validation.
  4. BERTScore F1 scoring of original vs back-translation to detect semantic drift.

Segments with BERTScore F1 < BERT_FLAG_THRESHOLD are flagged
(translation_flagged=True) but NOT excluded.

For testing, inject _pipeline_factory to avoid loading real HuggingFace models.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .models import SpeechSegment

logger = logging.getLogger(__name__)

BERT_FLAG_THRESHOLD: float = 0.85

_FWD_MODEL_TPL: str = "Helsinki-NLP/opus-mt-{src}-en"
_BWD_MODEL_TPL: str = "Helsinki-NLP/opus-mt-en-{src}"

# Languages for which back-translation models are unavailable or unreliable;
# these receive a fixed confidence score of 0.9 (above threshold).
_NO_BACK_TRANSLATION: frozenset[str] = frozenset({"zh", "ja", "ko", "ar", "hi"})

# Module-level imports with try/except so tests can patch
# vox_romana.emotion_flow.translator.detect
# vox_romana.emotion_flow.translator.bs_score
try:
    from langdetect import detect, LangDetectException as _LangDetectException
except ImportError:  # pragma: no cover
    detect = None  # type: ignore[assignment]
    _LangDetectException = Exception  # type: ignore[assignment,misc]

try:
    from bert_score import score as bs_score
except ImportError:  # pragma: no cover
    bs_score = None  # type: ignore[assignment]


def _default_pipeline_factory(task: str, model: str, **kwargs: Any) -> Any:
    """Load a HuggingFace pipeline. Separated for test injection."""
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(task, model=model, **kwargs)


class TranslatorAgent:
    """Translate speech segments to English and score back-translation quality.

    Args:
        cache_dir:          Optional directory for caching translation models.
        bert_threshold:     BERTScore F1 below which a segment is flagged.
        _force_passthrough: If True, skip all translation (for testing).
        _pipeline_factory:  Callable(task, model, **kwargs) -> pipeline.
                            Inject a mock for testing without real model loads.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        bert_threshold: float = BERT_FLAG_THRESHOLD,
        *,
        _force_passthrough: bool = False,
        _pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._bert_threshold = bert_threshold
        self._force_passthrough = _force_passthrough
        self._pipeline_factory = _pipeline_factory or _default_pipeline_factory
        self._pipeline_cache: dict[str, Any] = {}

    def translate(self, segments: list[SpeechSegment]) -> list[SpeechSegment]:
        """Translate segments to English and compute BERTScore confidence."""
        if self._force_passthrough or not segments:
            return [
                SpeechSegment(
                    text=s.original_text,
                    original_text=s.original_text,
                    source_lang="en",
                    translation_confidence=1.0,
                    translation_flagged=False,
                    segment_index=s.segment_index,
                    speech_id=s.speech_id,
                )
                for s in segments
            ]

        detected = [_detect_language(s.original_text) for s in segments]

        result: list[SpeechSegment] = []
        for seg, lang in zip(segments, detected):
            translated_text, confidence = self._translate_one(seg.original_text, lang)
            result.append(SpeechSegment(
                text=translated_text,
                original_text=seg.original_text,
                source_lang=lang,
                translation_confidence=confidence,
                translation_flagged=confidence < self._bert_threshold,
                segment_index=seg.segment_index,
                speech_id=seg.speech_id,
            ))
        return result

    def aggregate_quality(self, segments: list[SpeechSegment]) -> dict[str, Any]:
        """Summarise translation quality across a translated segment list."""
        if not segments:
            return {
                "n_total": 0,
                "n_flagged": 0,
                "mean_confidence": 1.0,
                "languages_seen": [],
            }
        n_flagged = sum(1 for s in segments if s.translation_flagged)
        mean_conf = sum(s.translation_confidence for s in segments) / len(segments)
        langs = sorted({s.source_lang for s in segments if s.source_lang})
        return {
            "n_total": len(segments),
            "n_flagged": n_flagged,
            "mean_confidence": mean_conf,
            "languages_seen": langs,
        }

    def _translate_one(self, text: str, lang: str) -> tuple[str, float]:
        """Return (english_text, bert_f1_score) for a single segment."""
        if lang == "en" or not lang:
            return text, 1.0

        fwd_model = _FWD_MODEL_TPL.format(src=lang)
        try:
            fwd_pipe = self._get_pipeline("translation", fwd_model)
            translated = fwd_pipe(text, max_length=512)[0]["translation_text"]
        except Exception as exc:
            logger.warning("Forward translation failed (%s→en): %s", lang, exc)
            return text, 0.0

        if lang in _NO_BACK_TRANSLATION:
            return translated, 0.9

        bwd_model = _BWD_MODEL_TPL.format(src=lang)
        try:
            bwd_pipe = self._get_pipeline("translation", bwd_model)
            back = bwd_pipe(translated, max_length=512)[0]["translation_text"]
        except Exception as exc:
            logger.warning("Back-translation failed (en→%s): %s", lang, exc)
            return translated, 0.8

        bert_f1 = _bert_score_f1([text], [back], lang=lang)
        return translated, bert_f1

    def _get_pipeline(self, task: str, model: str) -> Any:
        if model not in self._pipeline_cache:
            kwargs: dict[str, Any] = {"device": -1}
            if self._cache_dir:
                kwargs["model_kwargs"] = {"cache_dir": self._cache_dir}
            self._pipeline_cache[model] = self._pipeline_factory(
                task, model=model, **kwargs
            )
        return self._pipeline_cache[model]


def _detect_language(text: str) -> str:
    """Detect ISO 639-1 language code. Returns '' on failure."""
    if detect is None:
        return ""
    try:
        return detect(text)
    except Exception:
        return ""


def _bert_score_f1(refs: list[str], hyps: list[str], lang: str) -> float:
    """Compute BERTScore F1 between reference and hypothesis lists.

    Returns mean F1 as a float in [0, 1]. Returns 0.8 on failure.
    """
    if bs_score is None:
        return 0.8
    try:
        _, _, F = bs_score(hyps, refs, lang=lang, verbose=False, rescale_with_baseline=False)
        return float(F.mean().item())
    except Exception as exc:
        logger.warning("BERTScore computation failed: %s", exc)
        return 0.8
