"""Tests for SeminarPlan.suggest() — pure math, no LLM calls."""
from __future__ import annotations

import pytest

from vox_romana.university.config import UniversityConfig
from vox_romana.university.seminar_plan import SeminarPlan, _resolve_latencies


def _cfg(**kwargs) -> UniversityConfig:
    return UniversityConfig(**kwargs)


# ── latency resolution ────────────────────────────────────────────────────────

class TestResolveLatencies:
    def test_fallback_when_no_table(self):
        cfg = _cfg()
        lat = _resolve_latencies(cfg)
        assert lat["article"] == 15.0
        assert lat["synth"] == 40.0
        assert lat["exchange"] == 60.0

    def test_exact_model_match(self):
        cfg = _cfg(
            synthesis_model="deepseek-r1:14b",
            reading_model="deepseek-r1:7b",
            llm_latencies={
                "deepseek-r1:14b": {"article": 25, "synth": 60, "exchange": 90},
                "deepseek-r1:7b":  {"article": 8,  "synth": 20, "exchange": 15},
            },
        )
        lat = _resolve_latencies(cfg)
        assert lat["article"] == 25.0   # from 14b (synthesis model)
        assert lat["synth"] == 60.0     # from 14b (synthesis model)
        assert lat["exchange"] == 90.0  # from 14b (synthesis model — exchanges use synthesis_client)

    def test_default_row_fallback(self):
        cfg = _cfg(
            synthesis_model="some-unknown-model",
            llm_latencies={"default": {"article": 10, "synth": 30, "exchange": 50}},
        )
        lat = _resolve_latencies(cfg)
        assert lat["article"] == 10.0
        assert lat["synth"] == 30.0
        assert lat["exchange"] == 50.0

    def test_partial_row_merges_with_fallback(self):
        cfg = _cfg(
            synthesis_model="deepseek-r1:7b",
            llm_latencies={"deepseek-r1:7b": {"article": 8}},
        )
        lat = _resolve_latencies(cfg)
        assert lat["article"] == 8.0
        # unspecified keys fall through to module fallback
        assert lat["synth"] == 40.0
        assert lat["exchange"] == 60.0

    def test_bad_value_ignored(self):
        cfg = _cfg(
            synthesis_model="m",
            llm_latencies={"m": {"article": "oops", "synth": 30, "exchange": 50}},
        )
        lat = _resolve_latencies(cfg)
        assert lat["article"] == 15.0  # fallback
        assert lat["synth"] == 30.0


# ── SeminarPlan.suggest() ─────────────────────────────────────────────────────

class TestSeminarPlanSuggest:
    def _plan(self, budget_hours=0.5, n_scholars=3, **overrides) -> SeminarPlan:
        cfg = _cfg(
            llm_latencies={
                "deepseek-r1:14b": {"article": 25, "synth": 60, "exchange": 90}
            }
        )
        return SeminarPlan.suggest(
            budget_hours=budget_hours,
            n_scholars=n_scholars,
            config=cfg,
            **overrides,
        )

    def test_returns_seminar_plan(self):
        plan = self._plan()
        assert isinstance(plan, SeminarPlan)

    def test_all_derived_params_positive(self):
        plan = self._plan()
        assert plan.notes_per_synthesis >= 1
        assert plan.colloquium_every >= 1
        assert plan.colloquium_max_rounds >= 1

    def test_cli_override_notes_per_synthesis(self):
        plan = self._plan(notes_per_synthesis=7)
        assert plan.notes_per_synthesis == 7
        assert "notes_per_synthesis" in plan.overrides

    def test_cli_override_colloquium_every(self):
        plan = self._plan(colloquium_every=3)
        assert plan.colloquium_every == 3
        assert "colloquium_every" in plan.overrides

    def test_cli_override_colloquium_max_rounds(self):
        plan = self._plan(colloquium_max_rounds=5)
        assert plan.colloquium_max_rounds == 5
        assert "colloquium_max_rounds" in plan.overrides

    def test_overrides_not_in_dict_when_not_supplied(self):
        plan = self._plan()
        assert plan.overrides == {}

    def test_tight_budget_still_valid(self):
        # 5-minute budget — must not produce zeros or negatives
        plan = self._plan(budget_hours=0.083, n_scholars=3)
        assert plan.notes_per_synthesis >= 1
        assert plan.colloquium_every >= 1
        assert plan.colloquium_max_rounds >= 1

    def test_generous_budget_increases_intervals(self):
        tight = self._plan(budget_hours=0.083)
        generous = self._plan(budget_hours=4.0)
        # More time → more articles before sync (or at least as many)
        assert generous.notes_per_synthesis >= tight.notes_per_synthesis

    def test_more_scholars_reduces_max_rounds(self):
        few = self._plan(n_scholars=2)
        many = self._plan(n_scholars=5)
        # More scholars = more exchanges per round = fewer rounds fit in budget
        assert few.colloquium_max_rounds >= many.colloquium_max_rounds

    def test_budget_secs_property(self):
        plan = self._plan(budget_hours=1.0)
        assert plan.budget_secs == pytest.approx(3600.0)

    def test_summary_is_string(self):
        assert isinstance(self._plan().summary(), str)

    def test_summary_contains_key_fields(self):
        plan = self._plan(budget_hours=0.5)
        s = plan.summary()
        assert "0.500" in s or "0.5" in s
        assert "notes_per_synth" in s
        assert "colloquium_every" in s
        assert "max_rounds" in s
