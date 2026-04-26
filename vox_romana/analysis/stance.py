"""LLM-based stance extraction for SpeakerCorpus objects.

For each LDA topic in a fitted :class:`~vox_romana.analysis.topics.TopicModeler`,
the extractor sends the topic's top terms plus a sample of high-weight
utterances to DeepSeek and asks it to classify the speaker's stance as:

    ``pro``       — speaker supports / endorses the topic's position
    ``anti``      — speaker opposes / criticises the topic's position
    ``neutral``   — speaker discusses the topic without a clear lean
    ``ambiguous`` — insufficient evidence to determine stance

A :class:`~vox_romana.analysis.models.StanceRecord` is produced for every
(speaker, topic) pair whose mean topic weight exceeds ``min_weight``.

GRACEFUL DEGRADATION
    If no ``DEEPSEEK_API_KEY`` is found, or if the LLM call fails for any
    reason, the extractor falls back to ``position="ambiguous"`` with
    ``confidence=0.0``.  The rest of the pipeline continues unaffected.

USAGE::

    from vox_romana.analysis.stance import StanceExtractor
    from vox_romana.analysis.topics import TopicModeler

    tm = TopicModeler(n_topics=8)
    dist_a, dist_b = tm.fit_transform(corpus_a, corpus_b)

    stances = StanceExtractor().extract(
        corpus_a, corpus_b, tm, dist_a, dist_b
    )
    # list[StanceRecord] — one entry per (speaker, topic) above min_weight
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from .models import SpeakerCorpus, StanceRecord
from .topics import TopicModeler

logger = logging.getLogger(__name__)

_VALID_POSITIONS = {"pro", "anti", "neutral", "ambiguous"}
_POSITION_RE = re.compile(
    r'"position"\s*:\s*"(pro|anti|neutral|ambiguous)"', re.IGNORECASE
)
_SUMMARY_RE = re.compile(r'"summary"\s*:\s*"([^"]{5,})"')
_CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*([0-9.]+)')

_SYSTEM_PROMPT = """\
You are a neutral political science analyst.
You will be given a topic (described by key terms) and a set of utterances from a single speaker.
Classify the speaker's stance on this topic.

Respond with a JSON object containing exactly these keys:
  "position"   : one of "pro", "anti", "neutral", "ambiguous"
  "summary"    : one or two sentences summarising the stance
  "confidence" : a float 0.0–1.0 indicating how confident you are

Only output the JSON object. Do not add markdown fences or any extra text.
"""

_USER_TEMPLATE = """\
Topic key terms: {terms}

Speaker: {speaker}

Sample utterances (highest relevance to topic):
{utterances}

Classify {speaker}'s stance on the topic described by these terms.
"""


class StanceExtractor:
    """Extracts speaker stances on LDA topics via LLM.

    Args:
        min_weight:       Minimum mean topic weight for a topic to be
                          included in stance extraction (0.0–1.0).
                          Default 0.05 — excludes near-zero topics.
        n_sample_utts:    Number of high-weight utterances to send to the
                          LLM as evidence.  Default 5.
        temperature:      LLM temperature.  Keep low (0.0–0.2) for
                          reproducible stance classification.
        _llm_client:      Optional pre-initialised :class:`DeepSeekClient`
                          for test injection.  If ``None``, one is created
                          lazily when the first LLM call is needed.
    """

    def __init__(
        self,
        *,
        min_weight: float = 0.05,
        n_sample_utts: int = 5,
        temperature: float = 0.1,
        _llm_client: Any | None = None,
    ) -> None:
        self.min_weight = min_weight
        self.n_sample_utts = n_sample_utts
        self.temperature = temperature
        self._client = _llm_client
        self._llm_available: bool | None = None   # None = unknown

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def extract(
        self,
        corpus_a: SpeakerCorpus,
        corpus_b: SpeakerCorpus,
        topic_modeler: TopicModeler,
        dist_a,
        dist_b,
    ) -> list[StanceRecord]:
        """Extract stance records for both corpora.

        Args:
            corpus_a, corpus_b:  Speaker corpora.
            topic_modeler:       Fitted :class:`TopicModeler` (must be fitted).
            dist_a, dist_b:      Per-utterance topic distributions from
                                 ``topic_modeler.fit_transform()``.

        Returns:
            List of :class:`~vox_romana.analysis.models.StanceRecord` objects,
            one per (speaker, topic) pair whose mean weight ≥ ``min_weight``.
            Empty list if no topics qualify or LLM is unavailable.
        """
        import numpy as np

        topic_modeler._require_fitted()
        records: list[StanceRecord] = []

        n_topics = dist_a.shape[1]

        for topic_id in range(n_topics):
            label = topic_modeler.topic_label(topic_id)
            terms = topic_modeler.top_terms(topic_id, n_terms=8)

            for corpus, dist in (
                (corpus_a, dist_a),
                (corpus_b, dist_b),
            ):
                mean_w = float(np.mean(dist[:, topic_id]))
                if mean_w < self.min_weight:
                    logger.debug(
                        "Skipping %s / topic %d (weight=%.3f < %.3f)",
                        corpus.name, topic_id, mean_w, self.min_weight,
                    )
                    continue

                # Select utterances with highest weight on this topic
                weights = dist[:, topic_id]
                top_idx = np.argsort(weights)[::-1][: self.n_sample_utts]
                sample_utts = [corpus.utterances[i] for i in top_idx]

                record = self._classify(corpus, topic_id, label, terms, sample_utts)
                records.append(record)

        logger.info(
            "StanceExtractor: extracted %d stance records (%s, %s)",
            len(records),
            corpus_a.name,
            corpus_b.name,
        )
        return records

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _classify(
        self,
        corpus: SpeakerCorpus,
        topic_id: int,
        label: str,
        terms: list[str],
        sample_utts: list[str],
    ) -> StanceRecord:
        """Ask the LLM to classify stance; fall back on failure."""
        position = "ambiguous"
        summary = "Insufficient evidence to determine stance."
        confidence = 0.0
        evidence = sample_utts[0] if sample_utts else ""

        if self._ensure_llm():
            try:
                position, summary, confidence = self._llm_classify(
                    corpus.name, terms, sample_utts
                )
                evidence = sample_utts[0] if sample_utts else ""
            except Exception as exc:
                logger.warning(
                    "StanceExtractor LLM error for %s / topic %d: %s",
                    corpus.name, topic_id, exc,
                )

        return StanceRecord(
            speaker=corpus.name,
            topic_id=topic_id,
            topic_label=label,
            position=position,
            summary=summary,
            confidence=confidence,
            evidence_quote=evidence,
        )

    def _llm_classify(
        self, speaker: str, terms: list[str], sample_utts: list[str]
    ) -> tuple[str, str, float]:
        """Call DeepSeek and parse the JSON response.

        Returns:
            (position, summary, confidence)

        Raises:
            Exception: Propagated to :meth:`_classify` which handles fallback.
        """
        utterances_block = "\n".join(f"- {u}" for u in sample_utts)
        user_msg = _USER_TEMPLATE.format(
            terms=", ".join(terms),
            speaker=speaker,
            utterances=utterances_block,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        raw = self._client.chat(messages, temperature=self.temperature, max_tokens=256)
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> tuple[str, str, float]:
        """Parse LLM JSON response; fall back to regex on malformed output."""
        # Try clean JSON parse first
        try:
            # Strip markdown code fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
            data = json.loads(text)
            position = str(data.get("position", "ambiguous")).lower()
            if position not in _VALID_POSITIONS:
                position = "ambiguous"
            summary = str(data.get("summary", "")).strip() or "No summary provided."
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            return position, summary, confidence
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Regex fallback for partially-formed responses
        pos_m = _POSITION_RE.search(raw)
        sum_m = _SUMMARY_RE.search(raw)
        con_m = _CONFIDENCE_RE.search(raw)
        position = pos_m.group(1).lower() if pos_m else "ambiguous"
        summary = sum_m.group(1) if sum_m else "Unable to parse LLM summary."
        confidence = float(con_m.group(1)) if con_m else 0.3
        confidence = max(0.0, min(1.0, confidence))
        return position, summary, confidence

    def _ensure_llm(self) -> bool:
        """Lazily initialise DeepSeekClient.  Returns True if available."""
        if self._llm_available is not None:
            return self._llm_available

        if self._client is not None:
            self._llm_available = True
            return True

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            logger.info(
                "StanceExtractor: DEEPSEEK_API_KEY not set; "
                "all stances will be 'ambiguous'."
            )
            self._llm_available = False
            return False

        try:
            from vox_romana.core.client import DeepSeekClient
            self._client = DeepSeekClient(api_key=api_key)
            self._llm_available = True
        except Exception as exc:
            logger.warning("StanceExtractor: could not init DeepSeekClient: %s", exc)
            self._llm_available = False

        return self._llm_available
