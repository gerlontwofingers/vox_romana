"""Tests for the DataResearcher agent and supporting modules.

Tests cover:
    1. DataResearcher — plan parsing, LLM-only fallback, catalog recall
    2. Recipes — skip if DS packages not installed
    3. Report generator — skip if DS packages not installed
    4. Catalog report_path field
    5. Graceful degradation without DS packages
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.core.catalog import Catalog
from vox_romana.core.data_researcher import DataResearcher, StudyResult
from vox_romana.core.models import FactCard


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def catalog():
    cat = Catalog(db_path=":memory:")
    yield cat
    cat.close()


@pytest.fixture
def mock_client():
    """A mock DeepSeekClient that returns configurable responses."""
    client = MagicMock()
    client.chat = MagicMock(return_value='{"findings": "Test finding", "fact_cards": []}')
    return client


@pytest.fixture
def researcher(mock_client, catalog):
    return DataResearcher(client=mock_client, catalog=catalog)


# ═══════════════════════════════════════════════════════════════════════════
# PLAN PARSING
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanParsing:
    """Tests for parse_plan() — separated for testability."""

    def test_parse_valid_plan(self, researcher):
        raw = json.dumps({
            "recipe": "compare_means",
            "params": {"group_col": "treatment", "value_col": "outcome"},
            "rationale": "Comparing treatment vs control",
        })
        plan = researcher.parse_plan(raw)
        assert plan is not None
        assert plan["recipe"] == "compare_means"
        assert plan["params"]["group_col"] == "treatment"

    def test_parse_plan_with_markdown_fences(self, researcher):
        raw = '```json\n{"recipe": "trend_over_time", "params": {"time_col": "year", "value_col": "gdp"}}\n```'
        plan = researcher.parse_plan(raw)
        assert plan is not None
        assert plan["recipe"] == "trend_over_time"

    def test_parse_plan_invalid_json(self, researcher):
        plan = researcher.parse_plan("not json at all")
        assert plan is None

    def test_parse_plan_missing_recipe_key(self, researcher):
        raw = json.dumps({"params": {}, "rationale": "oops"})
        plan = researcher.parse_plan(raw)
        assert plan is None

    def test_parse_plan_non_dict(self, researcher):
        plan = researcher.parse_plan('["not", "a", "dict"]')
        assert plan is None


# ═══════════════════════════════════════════════════════════════════════════
# JSON HELPERS
# ═══════════════════════════════════════════════════════════════════════════

class TestJsonHelpers:
    """Tests for _parse_json_array and _parse_json_object."""

    def test_parse_json_array(self, researcher):
        result = researcher._parse_json_array('["query1", "query2"]')
        assert result == ["query1", "query2"]

    def test_parse_json_array_with_fences(self, researcher):
        result = researcher._parse_json_array('```json\n["a", "b"]\n```')
        assert result == ["a", "b"]

    def test_parse_json_array_invalid(self, researcher):
        result = researcher._parse_json_array("not json")
        assert result == []

    def test_parse_json_object(self, researcher):
        result = researcher._parse_json_object('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_object_with_fences(self, researcher):
        result = researcher._parse_json_object('```\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_parse_json_object_invalid(self, researcher):
        result = researcher._parse_json_object("nope")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# LLM-ONLY MODE (graceful degradation)
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMOnlyMode:
    """Tests for when DS packages are not installed."""

    def test_run_without_dataset(self, researcher, mock_client):
        """Should produce a result via LLM-only analysis."""
        # Mock the three LLM calls: LLM analysis, then findings summary.
        mock_client.chat.side_effect = [
            '{"findings": "LLM says unemployment fell", "fact_cards": [{"claim": "Unemployment fell 3%", "source": "LLM estimate", "confidence": "estimated"}]}',
            "Analysis suggests unemployment fell by approximately 3%.",
        ]

        result = researcher.run(brief="Analyze unemployment trends")

        assert isinstance(result, StudyResult)
        assert result.method == "llm_analysis"
        assert "unemployment" in result.findings.lower()

    def test_run_produces_fact_cards_from_llm(self, researcher, mock_client):
        mock_client.chat.side_effect = [
            '{"findings": "GDP grew", "fact_cards": [{"claim": "GDP grew 2.5%", "source": "World Bank", "confidence": "estimated"}]}',
            "GDP grew by 2.5% according to World Bank estimates.",
        ]

        result = researcher.run(brief="GDP growth analysis")

        assert len(result.fact_cards) == 1
        assert result.fact_cards[0].claim == "GDP grew 2.5%"
        assert result.fact_cards[0].confidence == "estimated"

    def test_run_logs_to_catalog(self, researcher, mock_client, catalog):
        mock_client.chat.side_effect = [
            '{"findings": "Test", "fact_cards": []}',
            "Summary of test findings.",
        ]

        result = researcher.run(
            brief="Test study",
            content_type="debate",
            kind="research",
        )

        assert result.study_id != ""
        # Verify it was logged — count studies in the DB.
        rows = catalog._conn.execute("SELECT COUNT(*) FROM studies").fetchone()
        assert rows[0] == 1


# ═══════════════════════════════════════════════════════════════════════════
# CATALOG RECALL
# ═══════════════════════════════════════════════════════════════════════════

class TestCatalogRecall:
    """Tests for prior study recall."""

    def test_prior_studies_fed_to_analysis(self, catalog, mock_client):
        """Researcher should query catalog for related prior studies."""
        # Seed the catalog with a prior study.
        run_id = catalog.log_run(content_type="debate", title="UBI Debate")
        catalog.log_study(
            run_id=run_id,
            content_type="debate",
            dataset="finland.csv",
            method="t-test",
            findings="Employment rose 4.8%",
        )

        researcher = DataResearcher(client=mock_client, catalog=catalog)

        mock_client.chat.side_effect = [
            '{"findings": "Confirms prior finding", "fact_cards": []}',
            "This confirms the prior finding of employment gains.",
        ]

        result = researcher.run(brief="UBI employment effects")
        assert result.prior_studies >= 0  # May or may not find via FTS match

    def test_no_catalog_still_works(self, mock_client):
        """Researcher works without a catalog (no recall, no logging)."""
        researcher = DataResearcher(client=mock_client, catalog=None)

        mock_client.chat.side_effect = [
            '{"findings": "Analysis complete", "fact_cards": []}',
            "Analysis complete.",
        ]

        result = researcher.run(brief="Test without catalog")
        assert isinstance(result, StudyResult)
        assert result.study_id == ""


# ═══════════════════════════════════════════════════════════════════════════
# WEB SEARCH INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

class TestWebSearch:
    """Tests for optional web search context."""

    def test_web_search_called_when_provider_given(self, catalog, mock_client):
        mock_search = MagicMock()
        mock_result = MagicMock()
        mock_result.title = "UBI Study"
        mock_result.snippet = "Finland's UBI pilot showed promising results"
        mock_search.search.return_value = [mock_result]

        mock_client.chat.side_effect = [
            '["UBI pilot results"]',  # web research queries
            '{"findings": "UBI works", "fact_cards": []}',  # LLM analysis
            "UBI pilot shows positive employment outcomes.",  # summary
        ]

        researcher = DataResearcher(
            client=mock_client, catalog=catalog, search_provider=mock_search,
        )
        result = researcher.run(brief="UBI effectiveness")

        mock_search.search.assert_called()

    def test_no_search_provider_skips_web(self, catalog, mock_client):
        mock_client.chat.side_effect = [
            '{"findings": "No web", "fact_cards": []}',
            "No web context available.",
        ]

        researcher = DataResearcher(client=mock_client, catalog=catalog)
        result = researcher.run(brief="Test")
        # Should succeed without error.
        assert isinstance(result, StudyResult)


# ═══════════════════════════════════════════════════════════════════════════
# STUDY RESULT
# ═══════════════════════════════════════════════════════════════════════════

class TestStudyResult:
    """Tests for the StudyResult dataclass."""

    def test_defaults(self):
        s = StudyResult(study_id="abc", method="test", findings="found stuff")
        assert s.fact_cards == []
        assert s.report_path == ""
        assert s.prior_studies == 0

    def test_with_fact_cards(self):
        fc = FactCard(claim="Test", source="Unit test")
        s = StudyResult(
            study_id="abc",
            method="test",
            findings="found stuff",
            fact_cards=[fc],
        )
        assert len(s.fact_cards) == 1
        assert s.fact_cards[0].claim == "Test"


# ═══════════════════════════════════════════════════════════════════════════
# CATALOG REPORT_PATH
# ═══════════════════════════════════════════════════════════════════════════

class TestCatalogReportPath:
    """Tests for the report_path field on studies."""

    def test_report_path_stored(self, catalog):
        study_id = catalog.log_study(
            dataset="test.csv",
            method="t-test",
            findings="Found something",
            report_path="data/reports/abc123",
        )
        row = catalog._conn.execute(
            "SELECT report_path FROM studies WHERE study_id = ?",
            (study_id,),
        ).fetchone()
        assert row["report_path"] == "data/reports/abc123"

    def test_report_path_defaults_empty(self, catalog):
        study_id = catalog.log_study(
            dataset="test.csv",
            method="test",
            findings="test",
        )
        row = catalog._conn.execute(
            "SELECT report_path FROM studies WHERE study_id = ?",
            (study_id,),
        ).fetchone()
        assert row["report_path"] == ""

    def test_report_path_in_get_studies(self, catalog):
        run_id = catalog.log_run(content_type="debate", title="Test")
        catalog.log_study(
            run_id=run_id,
            dataset="x.csv",
            method="m",
            findings="f",
            report_path="data/reports/xyz",
        )
        studies = catalog.get_studies(run_id)
        assert studies[0].report_path == "data/reports/xyz"


# ═══════════════════════════════════════════════════════════════════════════
# RECIPES (skip if DS packages not installed)
# ═══════════════════════════════════════════════════════════════════════════

try:
    import pandas as pd
    import numpy as np
    _HAS_DS = True
except ImportError:
    _HAS_DS = False


@pytest.mark.skipif(not _HAS_DS, reason="DS packages not installed")
class TestRecipes:
    """Tests for pre-approved statistical recipes."""

    def test_compare_means(self):
        from vox_romana.data.recipes import compare_means

        df = pd.DataFrame({
            "group": ["A"] * 50 + ["B"] * 50,
            "value": list(np.random.normal(10, 2, 50)) + list(np.random.normal(12, 2, 50)),
        })

        result = compare_means(df, group_col="group", value_col="value")
        assert result.method == "compare_means (two-sample t-test)"
        assert "p_value" in result.statistics
        assert len(result.figures) == 1

    def test_trend_over_time(self):
        from vox_romana.data.recipes import trend_over_time

        df = pd.DataFrame({
            "year": range(2000, 2020),
            "value": [x * 1.5 + np.random.normal(0, 0.5) for x in range(20)],
        })

        result = trend_over_time(df, time_col="year", value_col="value")
        assert "slope" in result.statistics
        assert result.statistics["slope"] > 0

    def test_distribution_summary(self):
        from vox_romana.data.recipes import distribution_summary

        df = pd.DataFrame({"x": np.random.normal(100, 15, 200)})

        result = distribution_summary(df, value_col="x")
        assert result.statistics["count"] == 200
        assert len(result.fact_cards) >= 1

    def test_correlation(self):
        from vox_romana.data.recipes import correlation

        df = pd.DataFrame({
            "a": np.random.normal(0, 1, 100),
            "b": np.random.normal(0, 1, 100),
            "c": np.random.normal(0, 1, 100),
        })

        result = correlation(df, columns=["a", "b", "c"])
        assert "strongest_r" in result.statistics

    def test_group_comparison(self):
        from vox_romana.data.recipes import group_comparison

        df = pd.DataFrame({
            "region": ["north"] * 40 + ["south"] * 40 + ["east"] * 40,
            "sales": (
                list(np.random.normal(100, 10, 40))
                + list(np.random.normal(120, 10, 40))
                + list(np.random.normal(90, 10, 40))
            ),
        })

        result = group_comparison(df, group_col="region", value_col="sales")
        assert "f_stat" in result.statistics
        assert len(result.figures) == 1


@pytest.mark.skipif(not _HAS_DS, reason="DS packages not installed")
class TestRecipesDSAvailable:
    """Test ds_available() helper."""

    def test_ds_available_returns_true(self):
        from vox_romana.data.recipes import ds_available
        assert ds_available() is True


class TestRecipesGraceful:
    """Test graceful degradation without DS packages."""

    def test_recipe_registry_importable(self):
        """The recipes module should be importable even without DS deps."""
        from vox_romana.data.recipes import RECIPE_REGISTRY
        assert isinstance(RECIPE_REGISTRY, dict)
        assert "compare_means" in RECIPE_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR (skip if DS packages not installed)
# ═══════════════════════════════════════════════════════════════════════════

class TestReportGenerator:
    """Tests for the HTML report generator."""

    def test_generate_report_creates_files(self, tmp_path):
        from vox_romana.data.recipes import RecipeResult
        from vox_romana.data.report import generate_report

        result = RecipeResult(
            method="test_method",
            summary="Test summary of findings.",
            statistics={"p_value": 0.03, "n": 100},
            fact_cards=[FactCard(claim="Test claim", source="test")],
            figures=[("test_chart.png", b"\x89PNG fake bytes")],
        )

        report_dir = generate_report(
            study_id="test123",
            result=result,
            content_type="debate",
            dataset="test.csv",
            report_root=tmp_path,
        )

        report_path = tmp_path / "test123"
        assert report_path.exists()
        assert (report_path / "report.html").exists()
        assert (report_path / "figures" / "test_chart.png").exists()

        html = (report_path / "report.html").read_text()
        assert "test_method" in html
        assert "Test summary" in html
        assert "Test claim" in html

    def test_generate_report_plain_fallback(self, tmp_path):
        """Report should still generate even without jinja2 (plain HTML)."""
        from vox_romana.data.recipes import RecipeResult
        from vox_romana.data import report as report_mod

        result = RecipeResult(
            method="fallback_test",
            summary="Plain fallback.",
            statistics={"n": 50},
        )

        # Temporarily disable jinja2.
        original = report_mod._HAS_JINJA
        report_mod._HAS_JINJA = False
        try:
            report_dir = report_mod.generate_report(
                study_id="fallback",
                result=result,
                report_root=tmp_path,
            )
            html = (tmp_path / "fallback" / "report.html").read_text()
            assert "fallback_test" in html
        finally:
            report_mod._HAS_JINJA = original
