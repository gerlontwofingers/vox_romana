"""Abstraction-ladder idea-flow tagging (shot 5.3-a).

The Speech Analyzer's net-new core: model *how a speaker moves between the
abstract/general and the concrete/specific*.  Each SENTENCE of a speaker's
corpus is classified into one of four abstraction buckets:

    ``abstract``  — general claims, principles, definitions, theory.
    ``mid``        — bridging statements: interpretation, framing, summary.
    ``concrete``   — specific facts, examples, data points, named things.
    ``anecdote``   — personal narrative, story, first-person experience.

The ordered sequence of these tags is the speaker's "idea-flow"; shot 5.3-b
turns it into a first-order transition matrix (the rhetorical fingerprint)
via the ``core.markov`` seam.

LLM-TAGGED (matching :class:`~vox_romana.analysis.stance.StanceExtractor`):
utterances are split into sentences and classified one batch at a time.
On missing API key or LLM failure, every sentence degrades to ``"mid"``
(the neutral bucket) with ``confidence=0.0`` so downstream code never sees
an empty flow.

USAGE::

    from vox_romana.analysis.abstraction import AbstractionTagger
    moves = AbstractionTagger().tag_corpus(corpus)
    # list[AbstractionMove] — one per sentence, in order
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from .models import SpeakerCorpus

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Move schema
# ─────────────────────────────────────────────────────────────────────────────

MOVE_LEVELS: tuple[str, ...] = ("abstract", "mid", "concrete", "anecdote")

_VALID_LEVELS = set(MOVE_LEVELS)


@dataclass(slots=True)
class AbstractionMove:
    """One sentence, tagged with its abstraction level.

    Attributes:
        speaker:     Speaker the sentence belongs to.
        sentence:    The original sentence text.
        level:       One of MOVE_LEVELS (``abstract``/``mid``/``concrete``/
                     ``anecdote``).
        confidence:  Tagger confidence 0.0–1.0 (0.0 on fallback).
        index:       Sentence index within the speaker's corpus (0-based).
    """

    speaker: str
    sentence: str
    level: str
    confidence: float
    index: int


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────

# Naive but adequate sentence boundary regex: split on .!? followed by
# whitespace + a capital letter (or end of string).  Handles the common
# abbreviations poorly, but matches the pragmatic bar of the analysis
# package (which favours determinism over linguistic perfection).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'0-9])")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentence strings (best-effort)."""
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a neutral rhetorical analyst.  You will be given a numbered list of \
sentences from a single speaker.  Classify each sentence by where it sits on \
the abstraction ladder:

  "abstract"  — general claims, principles, definitions, theory, big ideas.
  "mid"        — bridging: interpretation, framing, summarising, hedging.
  "concrete"   — specific facts, data, examples, named people/places/numbers.
  "anecdote"   — personal narrative, story, first-person experience.

Respond with a JSON object mapping each sentence index (as a string key) to \
an object with "level" and "confidence" (0.0–1.0).  Example:

{
  "0": {"level": "abstract", "confidence": 0.9},
  "1": {"level": "concrete", "confidence": 0.8}
}

Only output the JSON object. Do not add markdown fences or extra text.
"""

_USER_TEMPLATE = """\
Speaker: {speaker}

Sentences:
{sentences}

Classify each sentence by abstraction level.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Tagger
# ─────────────────────────────────────────────────────────────────────────────


class AbstractionTagger:
    """Classify a speaker corpus into sentence-level abstraction moves.

    Args:
        batch_size:   Number of sentences sent per LLM call.  Larger
                      batches mean fewer calls but longer prompts.  Default
                      24.
        temperature:  LLM temperature.  Keep low (0.0–0.2) for reproducible
                      classification.
        _llm_client:  Optional pre-initialised DeepSeekClient for test
                      injection.  Lazily created otherwise.
    """

    def __init__(
        self,
        *,
        batch_size: int = 24,
        temperature: float = 0.1,
        _llm_client: Any | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.temperature = temperature
        self._client = _llm_client
        self._llm_available: bool | None = None  # None = unknown

    # ── Public API ────────────────────────────────────────────────────────

    def tag_corpus(self, corpus: SpeakerCorpus) -> list[AbstractionMove]:
        """Tag every sentence of ``corpus`` and return ordered moves.

        Args:
            corpus: A :class:`~vox_romana.analysis.models.SpeakerCorpus`.

        Returns:
            One :class:`AbstractionMove` per sentence, in corpus order.
            Sentences are never dropped — on LLM failure each degrades to
            ``"mid"`` with ``confidence=0.0``.
        """
        # Flatten the corpus into (global_index, sentence) pairs.
        indexed: list[tuple[int, str]] = []
        for utt in corpus.utterances:
            for sentence in split_sentences(utt):
                indexed.append((len(indexed), sentence))

        if not indexed:
            logger.info("AbstractionTagger: corpus %r has no sentences.", corpus.name)
            return []

        # Tag in batches (LLM) or degrade wholesale (no LLM).
        tagged: dict[int, tuple[str, float]] = {}
        if self._ensure_llm():
            for start in range(0, len(indexed), self.batch_size):
                chunk = indexed[start : start + self.batch_size]
                try:
                    tagged.update(self._llm_tag(corpus.name, chunk))
                except Exception as exc:
                    logger.warning(
                        "AbstractionTagger LLM error for %s (batch %d..): %s",
                        corpus.name, start, exc,
                    )
                    # Degrade this batch; remaining sentences handled below.
        else:
            logger.info(
                "AbstractionTagger: DEEPSEEK_API_KEY not set; "
                "all moves will be 'mid' (confidence=0.0)."
            )

        moves: list[AbstractionMove] = []
        for idx, sentence in indexed:
            level, confidence = tagged.get(idx, ("mid", 0.0))
            moves.append(AbstractionMove(
                speaker=corpus.name,
                sentence=sentence,
                level=level,
                confidence=confidence,
                index=idx,
            ))
        logger.info(
            "AbstractionTagger: tagged %d sentences for %r.",
            len(moves), corpus.name,
        )
        return moves

    # ── LLM plumbing ──────────────────────────────────────────────────────

    def _llm_tag(
        self,
        speaker: str,
        chunk: list[tuple[int, str]],
    ) -> dict[int, tuple[str, float]]:
        """Tag one batch of sentences; returns ``{index: (level, confidence)}``."""
        lines = "\n".join(f"{idx}: {sentence}" for idx, sentence in chunk)
        user_msg = _USER_TEMPLATE.format(speaker=speaker, sentences=lines)
        raw = self._client.chat(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.temperature,
            max_tokens=512,
        )
        return self._parse_response(raw, {idx for idx, _ in chunk})

    def _parse_response(
        self,
        raw: str,
        expected_idx: set[int],
    ) -> dict[int, tuple[str, float]]:
        """Parse the LLM's ``{"0": {...}, "1": {...}}`` mapping."""
        result: dict[int, tuple[str, float]] = {}
        data = None
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            data = None

        if not isinstance(data, dict):
            # Fallback: give every expected index the neutral bucket.
            return {i: ("mid", 0.0) for i in expected_idx}

        for key, value in data.items():
            try:
                idx = int(str(key))
            except (TypeError, ValueError):
                continue
            if idx not in expected_idx:
                continue
            if isinstance(value, dict):
                level = str(value.get("level", "mid")).lower()
                if level not in _VALID_LEVELS:
                    level = "mid"
                try:
                    conf = float(value.get("confidence", 0.5))
                except (TypeError, ValueError):
                    conf = 0.5
                result[idx] = (level, max(0.0, min(1.0, conf)))

        # Any index the LLM didn't answer gets the neutral fallback.
        for i in expected_idx:
            result.setdefault(i, ("mid", 0.0))

        return result

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
                "AbstractionTagger: DEEPSEEK_API_KEY not set; "
                "all moves will be 'mid'."
            )
            self._llm_available = False
            return False

        try:
            from vox_romana.core.client import DeepSeekClient
            self._client = DeepSeekClient(api_key=api_key)
            self._llm_available = True
        except Exception as exc:
            logger.warning("AbstractionTagger: could not init DeepSeekClient: %s", exc)
            self._llm_available = False

        return self._llm_available
