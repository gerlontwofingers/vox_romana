"""EmotionTaggerAgent — classifies speech segments into 7 emotion categories.

Uses j-hartmann/emotion-english-distilroberta-base via HuggingFace transformers.
All 7 emotion scores are returned; the highest-scoring label is dominant.

For testing, inject _force_label (always returns that label with score 1.0)
or _pipeline_factory (replaces the transformers.pipeline loader).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .models import EMOTION_LABELS, EmotionTag, SpeechSegment

logger = logging.getLogger(__name__)

MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_BATCH_SIZE: int = 32


def _default_pipeline_factory(model: str, **kwargs: Any) -> Any:
    """Load a HuggingFace text-classification pipeline. Separated for test injection."""
    from transformers import pipeline as hf_pipeline
    return hf_pipeline("text-classification", model=model, **kwargs)


class EmotionTaggerAgent:
    """Tag speech segments with 7-class emotion scores.

    Args:
        cache_dir:         Optional HuggingFace model cache directory.
        batch_size:        Number of segments processed per pipeline call.
        _force_label:      If set, ALL segments receive this label (score 1.0,
                           all others 0.0). For testing only.
        _pipeline_factory: Callable(model, **kwargs) -> pipeline.
                           Inject a mock for testing without real model loads.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        *,
        _force_label: str | None = None,
        _pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._force_label = _force_label
        self._pipeline_factory = _pipeline_factory or _default_pipeline_factory
        self._pipeline: Any | None = None

    def tag(self, segments: list[SpeechSegment]) -> list[EmotionTag]:
        """Tag all segments. Returns a list parallel to `segments`."""
        if not segments:
            return []
        if self._force_label is not None:
            return [_forced_tag(self._force_label) for _ in segments]

        pipeline = self._get_pipeline()
        texts = [s.text for s in segments]
        tags: list[EmotionTag] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results = pipeline(batch, top_k=None)
            for result in results:
                tags.append(_parse_output(result))
        return tags

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            kwargs: dict[str, Any] = {"device": -1}
            if self._cache_dir:
                kwargs["model_kwargs"] = {"cache_dir": self._cache_dir}
            self._pipeline = self._pipeline_factory(MODEL_NAME, **kwargs)
        return self._pipeline


def _forced_tag(label: str) -> EmotionTag:
    """Return a tag where `label` has score 1.0 and all others 0.0."""
    if label not in EMOTION_LABELS:
        raise ValueError(
            f"Unknown emotion label: {label!r}. Must be one of {EMOTION_LABELS}."
        )
    scores = {l: 0.0 for l in EMOTION_LABELS}
    scores[label] = 1.0
    return EmotionTag(dominant=label, dominant_score=1.0, scores=scores)


def _parse_output(result: list[dict]) -> EmotionTag:
    """Convert pipeline output (list of {label, score}) to EmotionTag.

    The j-hartmann model returns lowercase labels matching EMOTION_LABELS.
    Lowercasing is applied defensively for forward-compatibility.
    """
    scores: dict[str, float] = {}
    for item in result:
        label = item["label"].lower()
        scores[label] = float(item["score"])

    # Fill any missing labels with 0.0
    for label in EMOTION_LABELS:
        scores.setdefault(label, 0.0)

    dominant = max(scores, key=scores.__getitem__)
    return EmotionTag(
        dominant=dominant,
        dominant_score=scores[dominant],
        scores={k: scores[k] for k in EMOTION_LABELS},
    )
