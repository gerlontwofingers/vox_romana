"""DataResearcher — a reusable agent that runs grounded data analysis.

The DataResearcher is show-agnostic. Debates, news shows, and any future
format can use it to get verified FactCards backed by real statistical analysis.

DEPENDENCY SPECS (for human review):
    Core DS:  requirements-data.txt  (pandas, scipy, sklearn, matplotlib, etc.)
    Deep DS:  requirements-data-deep.txt  (pytorch, tensorflow — optional)
    If DS packages are not installed, the agent degrades gracefully to
    text-only findings from the LLM (no charts or statistical verification).

FLOW:
    1. RECALL  — query catalog for related past studies (context, not truth)
    2. PLAN    — LLM picks a recipe + params from the recipe registry
    3. EXECUTE — run the controlled recipe on the provided data
    4. REPORT  — generate HTML report + FactCards
    5. LOG     — save to catalog (study + report_path)
    6. RETURN  — StudyResult with FactCards for downstream use

THE PRIOR STUDY PRINCIPLE:
    Before running any analysis, the DataResearcher searches the catalog
    for related past studies.  These are fed into the LLM's system prompt
    as context — NOT as ground truth.  The agent uses judgment to decide
    whether to build on, contradict, or ignore prior findings.

WEB ACCESS:
    The DataResearcher can use a SearchProvider (Google, Wikipedia, etc.)
    to gather reference context before analysis.  This is the same pattern
    used by DebateResearcher.  Web search is read-only and injectable.

SAFETY:
    - The agent does NOT write arbitrary code.  It picks from RECIPE_REGISTRY.
    - Recipes are pre-approved, tested functions in vox_romana/data/recipes/.
    - All DS execution happens inside recipe functions with controlled I/O.

EXAMPLE:
    from vox_romana.core.client import DeepSeekClient
    from vox_romana.core.catalog import Catalog
    from vox_romana.core.data_researcher import DataResearcher

    researcher = DataResearcher(
        client=DeepSeekClient(),
        catalog=Catalog(),
    )

    result = researcher.run(
        brief="Analyze employment outcomes from Finland's UBI pilot",
        dataset_path="data/finland_ubi_pilot.csv",
        content_type="debate",
    )

    for fc in result.fact_cards:
        print(f"  {fc.claim} (source: {fc.source})")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import DeepSeekClient
from .catalog import Catalog
from .models import FactCard
from .search import BaseSearchProvider

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports for optional DS packages.
# ─────────────────────────────────────────────────────────────────────────────

def _load_recipes() -> dict[str, Any] | None:
    """Try to import the recipe registry. Returns None if DS deps missing."""
    try:
        from ..data.recipes import RECIPE_REGISTRY, ds_available
        if not ds_available():
            return None
        return RECIPE_REGISTRY
    except ImportError:
        return None


def _load_report_generator():
    """Try to import the report generator."""
    try:
        from ..data.report import generate_report
        return generate_report
    except ImportError:
        return None


def _load_pandas():
    """Try to import pandas."""
    try:
        import pandas as pd
        return pd
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# StudyResult — what the DataResearcher returns.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StudyResult:
    """Complete output of a DataResearcher run.

    Attributes:
        study_id:     Catalog study ID (for linking).
        method:       Statistical method used.
        findings:     Plain English summary.
        fact_cards:   Verified FactCards for downstream use.
        report_path:  Path to the HTML report directory (if generated).
        prior_studies: Number of prior studies consulted.
    """

    study_id: str
    method: str
    findings: str
    fact_cards: list[FactCard] = field(default_factory=list)
    report_path: str = ""
    prior_studies: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """\
You are a data research planner for a media production system.
Given a research brief and a dataset description, you select the best
statistical recipe and parameters to answer the research question.

AVAILABLE RECIPES:
{recipe_menu}

PRIOR STUDIES (related past research — use your judgment on relevance):
{prior_studies}

RULES:
- Pick ONE recipe that best answers the research question.
- Fill in the required parameters based on the dataset columns.
- If the brief is ambiguous, pick the most informative analysis.
- Previous research is context, NOT truth. Build on it, contradict it,
  or ignore it as your judgment dictates.
- Respond with JSON ONLY, no explanation:

{{
    "recipe": "<recipe_name>",
    "params": {{<param_name>: <value>, ...}},
    "rationale": "<one sentence explaining your choice>"
}}
"""

FINDINGS_SYSTEM_PROMPT = """\
You are a research analyst summarizing statistical results for a media production.
Given raw statistics and any prior related research, write a clear, factual
summary in 2-4 sentences. This will be used to brief content producers.

RULES:
- State findings plainly. No jargon.
- Include key numbers with context.
- Note whether this confirms, contradicts, or extends prior research.
- Do NOT speculate beyond what the data shows.
"""

WEB_RESEARCH_PROMPT = """\
You are a research assistant. Given a data analysis brief, generate 2-3
search queries to find background context (definitions, prior studies,
methodology references) that will help interpret the results.

Respond with a JSON array of query strings ONLY.
"""


# ═══════════════════════════════════════════════════════════════════════════
# DataResearcher agent
# ═══════════════════════════════════════════════════════════════════════════

class DataResearcher:
    """Reusable data analysis agent with catalog recall and report generation.

    The DataResearcher follows the single-shot agent pattern used throughout
    vox_romana: dependency-inject the client, call run(), get structured output.

    It has four injectable dependencies:
        - client:          DeepSeekClient for LLM calls.
        - catalog:         Catalog for prior study recall + logging.
        - search_provider: Optional web search for background context.

    If DS packages (requirements-data.txt) are not installed, the agent
    degrades to LLM-only analysis — still useful, just no charts or
    statistical verification.
    """

    def __init__(
        self,
        client: DeepSeekClient,
        catalog: Catalog | None = None,
        search_provider: BaseSearchProvider | None = None,
        memory: Any | None = None,
    ) -> None:
        self._client = client
        self._catalog = catalog
        self._search = search_provider
        self._memory = memory  # MemoryRecall instance (optional)

    def run(
        self,
        brief: str,
        *,
        dataset_path: str | Path | None = None,
        content_type: str = "",
        kind: str = "research",
        run_id: str | None = None,
    ) -> StudyResult:
        """Run a data research study.

        Args:
            brief:        Natural language description of what to analyze.
            dataset_path: Path to a CSV/Parquet file (optional).
            content_type: Show type this research supports ("debate", "news").
            kind:         Study category (default "research").
            run_id:       Optional link to a content run in the catalog.

        Returns:
            StudyResult with findings, FactCards, and report path.
        """
        recipes = _load_recipes()
        pd = _load_pandas()

        # ── Step 1: Recall prior knowledge ────────────────────────────
        prior_context = ""
        prior_count = 0

        if self._memory is not None:
            # Use the learned memory system (semantic + LM synthesis).
            try:
                recall_result = self._memory.query(brief)
                prior_context = self._memory.format_context(recall_result)
                prior_count = len(recall_result.catalog_studies)
            except Exception as e:
                logger.warning("Memory recall failed, falling back to FTS5: %s", e)

        if not prior_context and self._catalog:
            # Fallback: raw FTS5 search on the catalog.
            prior_results = self._catalog.search(brief, limit=5)
            prior_studies = []
            for pr in prior_results:
                studies = self._catalog.get_studies(pr.run_id)
                prior_studies.extend(studies)

            if prior_studies:
                prior_count = len(prior_studies)
                lines = []
                for s in prior_studies[:10]:
                    lines.append(
                        f"- [{s.content_type}] {s.method}: {s.findings} "
                        f"(dataset: {s.dataset}, {s.created_at[:10]})"
                    )
                prior_context = "\n".join(lines)

        if not prior_context:
            prior_context = "(no prior studies found)"

        # ── Step 2: Optional web research for context ─────────────────
        web_context = ""
        if self._search:
            web_context = self._gather_web_context(brief)

        # ── Step 3: Load dataset (if provided and pandas available) ───
        df = None
        dataset_desc = ""
        if dataset_path and pd is not None:
            try:
                path = Path(dataset_path)
                if path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                dataset_desc = (
                    f"Columns: {', '.join(df.columns.tolist())}\n"
                    f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                    f"Dtypes:\n{df.dtypes.to_string()}\n"
                    f"First 3 rows:\n{df.head(3).to_string()}"
                )
            except Exception as e:
                logger.warning("Failed to load dataset %s: %s", dataset_path, e)
                dataset_desc = f"(failed to load: {e})"

        # ── Step 4: Plan — LLM picks recipe + params ─────────────────
        if recipes and df is not None:
            plan = self._plan_analysis(brief, dataset_desc, recipes, prior_context)
        else:
            plan = None

        # ── Step 5: Execute recipe OR fall back to LLM-only ──────────
        if plan and recipes:
            result = self._execute_recipe(df, plan, recipes)
        else:
            # No DS packages or no dataset — LLM-only analysis.
            result = self._llm_only_analysis(brief, prior_context, web_context)

        # ── Step 6: Generate findings summary via LLM ─────────────────
        findings = self._summarize_findings(brief, result, prior_context)

        # ── Step 7: Generate HTML report ──────────────────────────────
        report_path = ""
        generate_report = _load_report_generator()
        study_id = ""
        if self._catalog:
            # Pre-generate study_id so the report dir matches.
            import uuid
            study_id = uuid.uuid4().hex[:12]

        if generate_report and result.get("recipe_result"):
            try:
                report_path = generate_report(
                    study_id=study_id or "unnamed",
                    result=result["recipe_result"],
                    content_type=content_type,
                    dataset=str(dataset_path or ""),
                )
            except Exception as e:
                logger.warning("Report generation failed: %s", e)

        # ── Step 8: Log to catalog ────────────────────────────────────
        fact_cards = result.get("fact_cards", [])
        if self._catalog and study_id:
            self._catalog.log_study(
                run_id=run_id,
                content_type=content_type,
                kind=kind,
                dataset=str(dataset_path or ""),
                method=result.get("method", "llm_analysis"),
                findings=findings,
                fact_cards=[fc.to_dict() for fc in fact_cards],
                report_path=report_path,
            )

        return StudyResult(
            study_id=study_id,
            method=result.get("method", "llm_analysis"),
            findings=findings,
            fact_cards=fact_cards,
            report_path=report_path,
            prior_studies=prior_count,
        )

    # ─────────────────────────────────────────────────────────────────
    # Internal methods (each testable in isolation)
    # ─────────────────────────────────────────────────────────────────

    def _gather_web_context(self, brief: str) -> str:
        """Search the web for background context on the research topic."""
        raw = self._client.chat(
            messages=[
                {"role": "system", "content": WEB_RESEARCH_PROMPT},
                {"role": "user", "content": brief},
            ]
        )

        queries = self._parse_json_array(raw)
        if not queries:
            return ""

        snippets = []
        for q in queries[:3]:
            try:
                results = self._search.search(q, num_results=3)
                for r in results:
                    snippets.append(f"- {r.title}: {r.snippet}")
            except Exception:
                continue

        return "\n".join(snippets[:15])

    def _plan_analysis(
        self,
        brief: str,
        dataset_desc: str,
        recipes: dict[str, Any],
        prior_context: str,
    ) -> dict[str, Any] | None:
        """Have the LLM pick a recipe and parameters."""
        menu_lines = []
        for name, info in recipes.items():
            menu_lines.append(
                f"  {name}: {info['description']} "
                f"(params: {', '.join(info['params'])})"
            )
        recipe_menu = "\n".join(menu_lines)

        system = PLANNER_SYSTEM_PROMPT.format(
            recipe_menu=recipe_menu,
            prior_studies=prior_context,
        )

        user_prompt = f"BRIEF: {brief}\n\nDATASET:\n{dataset_desc}"

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
        )

        return self.parse_plan(raw)

    def parse_plan(self, raw_json: str) -> dict[str, Any] | None:
        """Parse the LLM's recipe selection response.

        Separated for testability.

        Returns:
            Dict with "recipe", "params", "rationale" keys, or None.
        """
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            plan = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse plan JSON: %.200s", raw_json)
            return None

        if not isinstance(plan, dict) or "recipe" not in plan:
            logger.warning("Plan missing 'recipe' key: %s", plan)
            return None

        return plan

    def _execute_recipe(
        self,
        df: Any,
        plan: dict[str, Any],
        recipes: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the planned recipe on the dataset."""
        recipe_name = plan["recipe"]
        params = plan.get("params", {})

        if recipe_name not in recipes:
            logger.warning("Unknown recipe '%s', falling back to LLM-only.", recipe_name)
            return {"method": "llm_analysis", "fact_cards": []}

        recipe_fn = recipes[recipe_name]["fn"]

        try:
            recipe_result = recipe_fn(df, **params)
            return {
                "method": recipe_result.method,
                "fact_cards": recipe_result.fact_cards,
                "statistics": recipe_result.statistics,
                "summary": recipe_result.summary,
                "recipe_result": recipe_result,
            }
        except Exception as e:
            logger.error("Recipe '%s' failed: %s", recipe_name, e)
            return {"method": f"{recipe_name} (failed)", "fact_cards": []}

    def _llm_only_analysis(
        self,
        brief: str,
        prior_context: str,
        web_context: str,
    ) -> dict[str, Any]:
        """Fall back to pure LLM analysis when DS packages aren't available."""
        context_parts = []
        if prior_context and prior_context != "(no prior studies found)":
            context_parts.append(f"PRIOR RESEARCH:\n{prior_context}")
        if web_context:
            context_parts.append(f"WEB CONTEXT:\n{web_context}")

        context = "\n\n".join(context_parts) if context_parts else "(no additional context)"

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": (
                    "You are a data analyst. Analyze the research question based "
                    "on available context. Produce findings as JSON:\n"
                    '{"findings": "...", "fact_cards": [{"claim": "...", '
                    '"source": "...", "confidence": "estimated"}]}'
                )},
                {"role": "user", "content": f"BRIEF: {brief}\n\n{context}"},
            ]
        )

        parsed = self._parse_json_object(raw)
        fact_cards = []
        if parsed and "fact_cards" in parsed:
            for fc_dict in parsed["fact_cards"]:
                fact_cards.append(FactCard(
                    claim=fc_dict.get("claim", ""),
                    source=fc_dict.get("source", "LLM analysis"),
                    confidence=fc_dict.get("confidence", "estimated"),
                ))

        return {
            "method": "llm_analysis",
            "fact_cards": fact_cards,
            "summary": parsed.get("findings", "") if parsed else "",
        }

    def _summarize_findings(
        self,
        brief: str,
        result: dict[str, Any],
        prior_context: str,
    ) -> str:
        """Have the LLM write a concise findings summary."""
        stats_summary = result.get("summary", "")
        if not stats_summary:
            stats_summary = json.dumps(
                result.get("statistics", {}), indent=2, default=str
            )

        user_prompt = (
            f"RESEARCH QUESTION: {brief}\n\n"
            f"RESULTS:\n{stats_summary}\n\n"
            f"PRIOR RESEARCH:\n{prior_context}"
        )

        return self._client.chat(
            messages=[
                {"role": "system", "content": FINDINGS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

    # ─────────────────────────────────────────────────────────────────
    # JSON parsing helpers
    # ─────────────────────────────────────────────────────────────────

    def _parse_json_array(self, raw: str) -> list[str]:
        """Parse a JSON array from LLM output, stripping markdown fences."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass
        return []

    def _parse_json_object(self, raw: str) -> dict[str, Any] | None:
        """Parse a JSON object from LLM output, stripping markdown fences."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        return None
