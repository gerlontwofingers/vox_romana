"""SeminarPlan — budget-aware parameter planner for University.seminar().

Given a wall-clock budget and a set of topics, ``SeminarPlan.suggest()``
derives the colloquium and synthesis pacing parameters so the seminar fits
within the budget without manual yaml tuning.

Resolution order (highest priority wins):
  1. Explicit CLI overrides passed to ``suggest()`` as keyword args.
  2. Math derived from ``budget_hours``, ``n_scholars``, and per-model
     latency estimates.
  3. ``UniversityConfig`` defaults (yaml values).

The latency estimates live in ``vox_romana.yaml`` under
``university.llm_latencies`` and are keyed by Ollama model tag::

    university:
      llm_latencies:
        "deepseek-r1:7b":  {article: 8,  synth: 20, exchange: 25}
        "deepseek-r1:14b": {article: 25, synth: 60, exchange: 90}
        default:           {article: 15, synth: 40, exchange: 60}

If no latency entry is found for the configured model the ``default`` row
is used; if that is also absent, hard-coded conservative values are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vox_romana.university.config import UniversityConfig

# Hard-coded fallback latencies (seconds) when nothing is configured.
_FALLBACK_LATENCIES: dict[str, float] = {
    "article": 15.0,
    "synth": 40.0,
    "exchange": 60.0,
}

# Budget fractions reserved for each phase (must sum ≤ 1.0).
_FRAC_READ = 0.55
_FRAC_SYNTH = 0.20
_FRAC_COLLO = 0.25


@dataclass
class SeminarPlan:
    """Resolved parameters for a single ``University.seminar()`` invocation.

    All fields are positive integers / positive floats unless noted.
    """

    # ── budget provenance ─────────────────────────────────────────────────────
    budget_hours: float
    n_scholars: int

    # ── per-model latency estimates used to derive the plan ───────────────────
    article_secs: float
    synth_secs: float
    exchange_secs: float

    # ── derived pacing parameters ─────────────────────────────────────────────
    notes_per_synthesis: int     # notes to accumulate before triggering synth
    colloquium_every: int        # articles per scholar between colloquium rounds
    colloquium_max_rounds: int   # upper bound on total colloquium rounds
    feasible_colloquium: bool    # False → budget too tight for even 1 full round

    # ── debug / audit ─────────────────────────────────────────────────────────
    overrides: dict = field(default_factory=dict, repr=False)

    # ── convenience ───────────────────────────────────────────────────────────
    @property
    def budget_secs(self) -> float:
        return self.budget_hours * 3600.0

    @classmethod
    def suggest(
        cls,
        *,
        budget_hours: float,
        n_scholars: int,
        config: "UniversityConfig",
        # explicit CLI overrides — None means "let the math decide"
        notes_per_synthesis: int | None = None,
        colloquium_every: int | None = None,
        colloquium_max_rounds: int | None = None,
    ) -> "SeminarPlan":
        """Derive a :class:`SeminarPlan` from the time budget and model latencies.

        Args:
            budget_hours:         Wall-clock budget supplied via ``--hours``.
            n_scholars:           Number of parallel scholars.
            config:               Active :class:`UniversityConfig`.
            notes_per_synthesis:  CLI override; ``None`` → derived from budget.
            colloquium_every:     CLI override; ``None`` → derived from budget.
            colloquium_max_rounds: CLI override; ``None`` → derived from budget.

        Returns:
            A fully-resolved :class:`SeminarPlan`.
        """
        lat = _resolve_latencies(config)
        article_secs: float = lat["article"]
        synth_secs: float = lat["synth"]
        exchange_secs: float = lat["exchange"]

        budget_secs = budget_hours * 3600.0
        overrides: dict[str, int] = {}

        # ── phase budgets ────────────────────────────────────────────────────
        read_budget_secs = budget_secs * _FRAC_READ
        synth_budget_secs = budget_secs * _FRAC_SYNTH
        collo_budget_secs = budget_secs * _FRAC_COLLO

        # Ollama serializes LLM calls even when scholars run in parallel threads.
        # Seed wall-clock: n_scholars × articles_per_scholar × article_secs.
        # Colloquium wall-clock: n*(n-1) exchanges × exchange_secs (GPU-serial).
        articles_per_scholar = max(1, int(read_budget_secs / (n_scholars * article_secs)))

        # Serialized colloquium round cost (all pairs, one GPU).
        exchanges_per_round = n_scholars * max(0, n_scholars - 1)
        collo_round_cost_secs = exchanges_per_round * exchange_secs

        # ── notes_per_synthesis ───────────────────────────────────────────────
        if notes_per_synthesis is not None:
            nps = max(1, notes_per_synthesis)
            overrides["notes_per_synthesis"] = nps
        else:
            # How many synth passes fit in the synth budget?
            synth_passes = max(1, synth_budget_secs / synth_secs)
            nps = max(1, math.ceil(articles_per_scholar / synth_passes))

        # ── colloquium_every ──────────────────────────────────────────────────
        if colloquium_every is not None:
            ce = max(1, colloquium_every)
            overrides["colloquium_every"] = ce
        else:
            # How many colloquium rounds fit in the colloquium budget
            # (using serialized GPU cost, not wall-clock with parallelism)?
            collo_rounds = max(1, collo_budget_secs / collo_round_cost_secs) if collo_round_cost_secs > 0 else 1
            ce = max(1, math.ceil(articles_per_scholar / collo_rounds))

        # ── colloquium_max_rounds ─────────────────────────────────────────────
        if colloquium_max_rounds is not None:
            cmr = max(1, colloquium_max_rounds)
            overrides["colloquium_max_rounds"] = cmr
        else:
            cmr = max(1, int(collo_budget_secs / collo_round_cost_secs)) if collo_round_cost_secs > 0 else 1

        # ── feasibility check ─────────────────────────────────────────────────
        seed_cost_secs = n_scholars * ce * article_secs
        feasible_colloquium = (seed_cost_secs + collo_round_cost_secs) <= budget_secs

        return cls(
            budget_hours=budget_hours,
            n_scholars=n_scholars,
            article_secs=article_secs,
            synth_secs=synth_secs,
            exchange_secs=exchange_secs,
            notes_per_synthesis=nps,
            colloquium_every=ce,
            colloquium_max_rounds=cmr,
            feasible_colloquium=feasible_colloquium,
            overrides=overrides,
        )

    def summary(self) -> str:
        """Human-readable one-liner for the plan banner."""
        ov = f" (overrides: {self.overrides})" if self.overrides else ""
        return (
            f"budget={self.budget_hours:.3f}h  scholars={self.n_scholars}  "
            f"notes_per_synth={self.notes_per_synthesis}  "
            f"colloquium_every={self.colloquium_every}  "
            f"max_rounds={self.colloquium_max_rounds}"
            f"{ov}"
        )


def _resolve_latencies(config: "UniversityConfig") -> dict[str, float]:
    """Return latency estimates (seconds) for the configured models.

    All three keys (``article``, ``synth``, ``exchange``) come from the
    synthesis_model row (14b) because ``_colloquium_exchange`` uses
    ``synthesis_client``.

    Lookup order: exact match → "default" → fallback.
    """
    table: dict = getattr(config, "llm_latencies", {}) or {}

    def _row(model: str) -> dict:
        row = table.get(model)
        if not isinstance(row, dict):
            row = table.get("default") or {}
        return row if isinstance(row, dict) else {}

    synth_row = _row(config.synthesis_model)

    base = dict(_FALLBACK_LATENCIES)
    for k in ("article", "synth", "exchange"):
        if k in synth_row:
            try:
                base[k] = float(synth_row[k])
            except (TypeError, ValueError):
                pass
    return base
