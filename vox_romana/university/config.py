"""UniversityConfig — configuration for the Scholar/University research system.

Loaded from the ``university:`` block in ``vox_romana.yaml``.  Follows the
same conventions as :class:`vox_romana.core.config.ContentConfig`:

* Mutable (not frozen) — values may be overridden programmatically after
  loading from YAML.
* ``validate()`` returns a list of problem strings rather than raising, so
  callers can decide whether to bail or proceed with warnings.

USAGE::

    from vox_romana.university.config import UniversityConfig

    cfg = UniversityConfig.from_yaml_block({"model": "deepseek-r1:14b"})
    problems = cfg.validate()
    if problems:
        for p in problems:
            print(f"  ⚠ {p}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UniversityConfig:
    """Configuration for the Scholar deep-research agent.

    All fields have sane defaults so an empty YAML block still produces a
    working config.  Override individual fields in ``vox_romana.yaml``:

    .. code-block:: yaml

        university:
          reading_model: deepseek-r1:7b
          synthesis_model: deepseek-r1:14b
          notes_per_synthesis: 5

    Attributes:
        reading_model: Ollama model tag used for per-article note extraction.
            A fast, lightweight model (e.g. ``deepseek-r1:7b``) keeps
            per-article latency low.
        synthesis_model: Ollama model tag used for synthesis, queue
            prioritisation, and final write-up.  A larger, higher-quality
            model (e.g. ``deepseek-r1:14b``) improves coherence here.
        base_url: Ollama API base URL.
        temperature: Sampling temperature for all Scholar LLM calls.
        timeout_secs: Per-request timeout in seconds. Long articles may
            need 5–10 minutes on a small model.
        notes_per_synthesis: How many new notes to accumulate before
            triggering a synthesis update. Lower = more frequent synthesis
            (slower); higher = coarser (faster).
        max_links_per_article: Maximum outbound links to add to the queue
            from a single article. Caps graph explosion.
        relevance_threshold: Minimum relevance score (0–10) for a link to
            be added to the queue. Links scoring below this are discarded.
        max_queue_size: Hard cap on the article queue. Prevents unbounded
            graph traversal on highly connected topics.
        wikipedia_cache_ttl_days: Cache freshness for WikipediaSource.
            ``0`` = always re-fetch. Negative = never expire.
        priority_interval: How many articles between queue re-prioritisations.
            Higher = less LLM overhead but potentially worse ordering.
        language: Wikipedia language code used by WikipediaSource.
    """

    reading_model: str = "deepseek-r1:7b"
    synthesis_model: str = "deepseek-r1:14b"
    base_url: str = "http://localhost:11434/v1"
    temperature: float = 0.6
    timeout_secs: int = 300
    notes_per_synthesis: int = 10
    max_links_per_article: int = 15
    relevance_threshold: float = 4.0
    max_queue_size: int = 500
    wikipedia_cache_ttl_days: float = 30.0
    priority_interval: int = 20
    language: str = "en"
    openalex_enabled: bool = True
    colloquium_enabled: bool = True
    colloquium_interval: int = 25
    colloquium_max_leads_per_peer: int = 3
    colloquium_snippet_chars: int = 1500
    # Per-model latency hints (seconds) used by SeminarPlan.suggest().
    # Keys are Ollama model tags; 'default' is the fallback.
    # Example entry: {"deepseek-r1:14b": {"article": 25, "synth": 60, "exchange": 90}}
    llm_latencies: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return a list of problems with this config. Empty list = all good.

        Mirrors the soft-check pattern from
        :meth:`vox_romana.core.config.ContentConfig.validate`.
        """
        problems: list[str] = []

        if not self.reading_model.strip():
            problems.append(
                "reading_model is empty — set university.reading_model in vox_romana.yaml."
            )

        if not self.synthesis_model.strip():
            problems.append(
                "synthesis_model is empty — set university.synthesis_model in vox_romana.yaml."
            )

        if not self.base_url.strip():
            problems.append(
                "base_url is empty — Ollama server URL required "
                "(default: http://localhost:11434/v1)."
            )

        if not (0.0 <= self.temperature <= 2.0):
            problems.append(
                f"temperature must be in [0.0, 2.0], got {self.temperature}."
            )

        if self.timeout_secs <= 0:
            problems.append(
                f"timeout_secs must be positive, got {self.timeout_secs}."
            )

        if self.notes_per_synthesis < 1:
            problems.append(
                f"notes_per_synthesis must be ≥ 1, got {self.notes_per_synthesis}."
            )

        if self.max_links_per_article < 0:
            problems.append(
                f"max_links_per_article must be ≥ 0, got {self.max_links_per_article}."
            )

        if not (0.0 <= self.relevance_threshold <= 10.0):
            problems.append(
                f"relevance_threshold must be in [0.0, 10.0], "
                f"got {self.relevance_threshold}."
            )

        if self.max_queue_size < 1:
            problems.append(
                f"max_queue_size must be ≥ 1, got {self.max_queue_size}."
            )

        return problems

    @classmethod
    def from_yaml_block(cls, block: dict[str, Any] | None) -> "UniversityConfig":
        """Build a UniversityConfig from the ``university:`` YAML dict.

        Unknown keys are silently ignored so future YAML additions don't
        break older code.  Missing keys fall through to the class defaults.

        Args:
            block: The ``university`` sub-dict from ``vox_romana.yaml``, or
                   ``None`` / ``{}`` to use all defaults.
        """
        if not block:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in block.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_project_yaml(cls) -> "UniversityConfig":
        """Load from the project-root ``vox_romana.yaml`` (convenience factory).

        Falls back to all defaults if the file or the ``university`` block is
        absent.
        """
        try:
            from vox_romana.core.paths import _load_config  # type: ignore[attr-defined]
            block = _load_config().get("university") or {}
        except Exception:
            block = {}
        return cls.from_yaml_block(block)
