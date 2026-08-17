"""The Research Agent — turns a brief into grounded reference material.

This is "Step 0" of the bootstrap chain. It runs BEFORE the Architect, and
its job is to answer the question: "What are the actual facts about this topic?"

WITHOUT the research agent:
    brief → Architect → Profile Builder → Director
    (participants have no factual claims — they argue from vibes)

WITH the research agent:
    brief → RESEARCHER → reference_material → Architect → Profile Builder → Director
    (participants get real statistics, study names, and verifiable claims)

THE TWO-STEP PROCESS:

    Step 1 — QUERY FORMULATION:
        The brief "Two economists debate UBI" becomes 3-5 targeted search queries:
          - "universal basic income economic research results"
          - "UBI pilot programs outcomes data"
          - "arguments against universal basic income economics"
          - "Finland UBI experiment 2017 results"

        Why multiple queries? Because a single query gives you one perspective.
        We deliberately search for BOTH sides of the debate topic so the
        reference material is balanced. Biased research → biased debate.

    Step 2 — COMPILATION:
        Search results (titles + snippets) get fed to the LLM, which extracts
        and organizes the key facts into a structured reference block:

          "## Key Statistics
           - Finland's 2017-2018 UBI pilot: 2,000 participants, €560/month...
           ## Pro Arguments (with sources)
           - Stockton SEED program showed 12% increase in full-time employment...
           ## Con Arguments (with sources)
           - A nationwide UBI at $1,000/month would cost ~$3.1 trillion..."

        This reference block becomes the `reference_material` that flows
        through the entire bootstrap chain.

SEARCH PROVIDER:
    The research agent doesn't care HOW you search the web. It accepts any
    BaseSearchProvider — Google (default), Tavily, Brave, or a mock for testing.
    See core/search.py for the interface.

TESTABILITY:
    - `formulate_queries()` is testable with canned JSON.
    - `compile_results()` is testable with canned search results.
    - `run()` chains them — mock the search provider + LLM client = fully testable.

EXAMPLE:
    from vox_romana.core.search import GoogleSearchProvider
    from vox_romana.core.client import DeepSeekClient
    from vox_romana.shows.debates.researcher import DebateResearcher

    researcher = DebateResearcher(
        client=DeepSeekClient(),
        search_provider=GoogleSearchProvider(),
    )

    reference_material = researcher.run("Should cities ban cars from downtown?")
    print(reference_material)
    # "## Key Statistics\n- Barcelona's superblocks reduced NO2 by 25%..."

COST:
    - 1 LLM call for query formulation (~$0.001)
    - 3-5 Google searches (free within 100/day quota)
    - 1 LLM call for compilation (~$0.002-0.005)
    - Total: ~$0.003-0.006 per debate, ~5-10 seconds
"""

from __future__ import annotations

from ...core.client import DeepSeekClient
from ...core.json_retry import JsonParseError, parse_json_array_response
from ...core.search import BaseSearchProvider, SearchResult


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

QUERY_FORMULATOR_PROMPT = """\
You are a research assistant preparing background material for a debate.
Given a debate brief, generate 3-5 targeted web search queries that will
find the best factual evidence for ALL sides of the topic.

RULES:
- Generate queries that cover BOTH sides (pro AND con). Balanced research
  produces balanced debates.
- Focus on: statistics, study results, named programs/policies, expert opinions.
- If a PERSPECTIVE CONTEXT block is provided, include location-specific or
    institution-specific queries that help ground the topic from that vantage
    point without suppressing counterarguments.
- Include at least one query specifically targeting counterarguments.
- Keep queries concise and search-engine-friendly (no full sentences).
- ALWAYS respond with a JSON array of strings. Nothing else.

EXAMPLE INPUT:
  "Two economists debate universal basic income"

EXAMPLE OUTPUT:
  [
    "universal basic income economic research results",
    "UBI pilot programs outcomes data statistics",
    "arguments against universal basic income economics",
    "Finland UBI experiment 2017 employment results",
    "cost estimates universal basic income United States"
  ]
"""

NEWS_QUERY_FORMULATOR_PROMPT = """\
You are a research assistant preparing background material for a show.
Given a show brief, generate 3-5 targeted search queries that will find
the best factual background information for the topics covered.

RULES:
- Focus on: current events, statistics, named programs/policies, expert opinions.
- If a PERSPECTIVE CONTEXT block is provided, include queries that surface
    facts, terminology, and institutions relevant to that vantage point.
- Keep queries concise and search-engine-friendly (no full sentences).
- Cover the main topics mentioned in the brief.
- ALWAYS respond with a JSON array of strings. Nothing else.

EXAMPLE INPUT:
  "Local news broadcast covering housing costs, public transit, and city budget"

EXAMPLE OUTPUT:
  [
    "local housing costs affordability data 2024",
    "public transit ridership funding statistics",
    "city budget allocations infrastructure 2024",
    "affordable housing policy programs local government",
    "transit expansion impact commuters"
  ]
"""

COMPILER_PROMPT = """\
You are a research compiler. You've been given search results about a debate
topic. Your job is to extract and organize the KEY FACTS into a structured
reference document that debate participants will use as their source material.

TOPIC: {topic}
{perspective_block}

SEARCH RESULTS:
{search_results}

RULES:
- Extract specific, verifiable facts: statistics, study names, dates, numbers.
- Organize into clear sections: Key Statistics, Pro Arguments (with sources),
  Con Arguments (with sources), Context/Background.
- Include the source (study name, organization, year) for every claim.
- Do NOT editorialize or add your own opinions.
- Do NOT invent facts that aren't in the search results.
- If perspective context is provided, keep observer-relative wording and
    institutional framing consistent with it.
- If the search results are thin on one side, note that explicitly.
- Use markdown formatting with ## headers and bullet points.
- Aim for 300-600 words — enough to ground the debate, not a dissertation.
"""

KNOWLEDGE_COMPILER_PROMPT = """\
You are a research compiler preparing background material for a debate.
No web search is available — compile what you KNOW from your training data.

TOPIC: {topic}
{perspective_block}

Your job: produce a structured reference document with REAL, VERIFIABLE facts
that debate participants will use as their source material.

RULES:
- Only include facts you are confident about. If you're unsure, say so
  explicitly (e.g., "approximately", "as of [last known data]").
- Include specific numbers, dates, study names, and sources where you
  can recall them. Approximate figures are fine if labeled as such.
- Cover BOTH sides — pro AND con. Balanced material produces balanced debates.
- Organize into clear sections: Key Statistics, Pro Arguments (with sources),
  Con Arguments (with sources), Context/Background.
- If perspective context is provided, keep observer-relative language and
    place naming consistent with that vantage point.
- For entertainment topics (TV shows, films, games), include: episode counts,
  plot-relevant specifics, notable events, fan-compiled data you're aware of.
- Do NOT fabricate precise statistics you don't actually know. Round numbers
  and qualitative summaries are honest; fake decimal points are not.
- Use markdown formatting with ## headers and bullet points.
- Aim for 400-800 words — be thorough, this is all the participants get.
"""

_PERSPECTIVE_CONTEXT_TEMPLATE = """\
PERSPECTIVE CONTEXT:
{perspective}
"""


# ═══════════════════════════════════════════════════════════════════════════
# DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

# How many search results to fetch per query.
DEFAULT_RESULTS_PER_QUERY = 5

# Maximum number of search queries to generate.
MAX_QUERIES = 5


class DebateResearcher:
    """Researches a debate topic and produces grounded reference material.

    Takes a brief, searches the web for evidence on all sides, and compiles
    the results into a structured reference block. This reference block
    becomes the `reference_material` that feeds into the bootstrap chain.

    Attributes:
        _client:          DeepSeekClient for LLM calls (query formulation + compilation).
        _search_provider: Any BaseSearchProvider (Google, Tavily, mock, etc.).
    """

    def __init__(
        self,
        client: DeepSeekClient,
        search_provider: BaseSearchProvider | None = None,
        query_formulator_prompt: str | None = None,
    ) -> None:
        """Initialize the researcher.

        Args:
            client:                   DeepSeekClient for LLM calls.
            search_provider:          Any search provider implementing BaseSearchProvider.
                                      In production: GoogleSearchProvider().
                                      In tests: a mock.
                                      If None, run() falls back to LLM-only research.
            query_formulator_prompt:  Override the system prompt used for query
                                      formulation. Defaults to the debate-specific
                                      QUERY_FORMULATOR_PROMPT. Pass
                                      NEWS_QUERY_FORMULATOR_PROMPT when using this
                                      researcher for a news show.
        """
        self._client = client
        self._search = search_provider
        self._query_formulator_prompt = query_formulator_prompt or QUERY_FORMULATOR_PROMPT

    def run(self, brief: str, *, perspective: str | None = None) -> str:
        """Research a topic and return structured reference material.

        This is the main entry point. With a search provider:
            brief → search queries → web search → raw results → compiled reference
        Without a search provider (LLM-only fallback):
            brief → LLM compiles from training knowledge

        Args:
            brief: The debate brief (natural language description).
            perspective: Optional explicit vantage point for the show.

        Returns:
            A structured reference material string (markdown-formatted)
            ready to be passed as `reference_material` to the bootstrap.

        Raises:
            ValueError: If query formulation returns unparseable JSON.
        """
        # ── LLM-only fallback: no search provider ───────────────────
        if self._search is None:
            return self.run_from_knowledge(brief, perspective=perspective)

        # ── Step 1: Formulate search queries ─────────────────────────
        queries = self.formulate_queries(brief, perspective=perspective)

        # ── Step 2: Search the web ───────────────────────────────────
        all_results: list[SearchResult] = []
        for query in queries:
            try:
                results = self._search.search(
                    query, num_results=DEFAULT_RESULTS_PER_QUERY
                )
                all_results.extend(results)
            except Exception:
                # If one query fails (rate limit, network, etc.), keep going.
                # Partial results are better than no results.
                continue

        # ── Step 3: Compile into reference material ──────────────────
        if not all_results:
            return ""  # No results → no reference material. Bootstrap continues without.

        return self.compile_results(brief, all_results, perspective=perspective)

    def formulate_queries(
        self,
        brief: str,
        *,
        perspective: str | None = None,
    ) -> list[str]:
        """Turn a brief into targeted search queries.

        One LLM call (with up to 2 retries on malformed JSON). The LLM reads
        the brief and produces 3-5 search queries designed to find balanced
        factual evidence.

        Separated from run() for testability — pass a canned brief, get
        deterministic queries back.

        Args:
            brief: The debate brief.
            perspective: Optional explicit vantage point for the show.

        Returns:
            A list of 3-5 search query strings.

        Raises:
            ValueError: If the LLM returns unparseable JSON after all retries.
        """
        user_content = f"BRIEF:\n{brief}"
        if perspective and perspective.strip():
            user_content = (
                f"{user_content}\n\n"
                f"{_PERSPECTIVE_CONTEXT_TEMPLATE.format(perspective=perspective.strip())}"
            )
        messages = [
            {"role": "system", "content": self._query_formulator_prompt},
            {"role": "user", "content": user_content},
        ]
        last_error: Exception | None = None

        for attempt in range(1, 4):  # up to 3 attempts total
            raw = self._client.chat(messages=messages, temperature=0)
            try:
                return self.parse_queries(raw)
            except ValueError as exc:
                last_error = exc
                feedback = (
                    f"Your previous response (attempt {attempt}) was not a valid "
                    f"JSON array. Respond ONLY with a JSON array of strings — "
                    f"no markdown fences, no explanation, no trailing commas. "
                    f"Example: [\"query one\", \"query two\"]"
                )
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": feedback},
                ]

        raise ValueError(
            f"Query formulator failed after 3 attempts. Last error: {last_error}"
        ) from last_error

    def parse_queries(self, raw_json: str) -> list[str]:
        """Parse the LLM's query formulation response.

        Testable without the LLM — pass a canned JSON string.

        Args:
            raw_json: Raw string from the LLM (should be a JSON array of strings).

        Returns:
            List of search query strings (capped at MAX_QUERIES).

        Raises:
            ValueError: If JSON is malformed.
        """
        try:
            queries = parse_json_array_response(raw_json)
        except JsonParseError as e:
            # Give a clearer error when the LLM returned a JSON object instead
            # of an array (common mistake: wraps queries in `{"queries": [...]}`).
            import json as _json
            try:
                parsed = _json.loads(raw_json.strip())
                if not isinstance(parsed, list):
                    raise ValueError(
                        f"Expected a JSON array of strings, got {type(parsed).__name__}"
                    )
            except _json.JSONDecodeError:
                pass
            raise ValueError(
                f"Query formulator returned invalid JSON. First 200 chars: "
                f"{raw_json[:200]!r}\nParse error: {e}"
            ) from e

        # Cap the number of queries to avoid burning through search quota.
        return [str(q) for q in queries[:MAX_QUERIES]]

    def compile_results(
        self,
        brief: str,
        results: list[SearchResult],
        *,
        perspective: str | None = None,
    ) -> str:
        """Compile search results into structured reference material.

        One LLM call. Feeds the raw search snippets to the LLM, which
        extracts key facts and organizes them into a clean reference doc.

        Args:
            brief:   The original debate brief (for context).
            results: List of SearchResult objects from web searches.

        Returns:
            Structured reference material (markdown string).
        """
        # ── Format search results for the LLM ────────────────────────
        # We include title + snippet for each result. URLs are included
        # for provenance but the LLM doesn't need to follow them.
        formatted_results = []
        seen_snippets: set[str] = set()  # Dedupe identical snippets.

        for r in results:
            # Skip results with no snippet (useless for fact extraction).
            if not r.snippet.strip():
                continue
            # Skip duplicate snippets (common when queries overlap).
            if r.snippet in seen_snippets:
                continue
            seen_snippets.add(r.snippet)

            formatted_results.append(
                f"TITLE: {r.title}\n"
                f"URL: {r.url}\n"
                f"SNIPPET: {r.snippet}"
            )

        if not formatted_results:
            return ""

        search_block = "\n\n".join(formatted_results)

        # ── Call the LLM to compile ──────────────────────────────────
        prompt = COMPILER_PROMPT.format(
            topic=brief,
            perspective_block=(
                f"{_PERSPECTIVE_CONTEXT_TEMPLATE.format(perspective=perspective.strip())}\n"
                if perspective and perspective.strip()
                else ""
            ),
            search_results=search_block,
        )

        return self._client.chat(
            messages=[
                {"role": "system", "content": "You are a factual research compiler."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

    def run_from_knowledge(
        self,
        brief: str,
        *,
        perspective: str | None = None,
    ) -> str:
        """Compile reference material from the LLM's training knowledge.

        Single LLM call. No web search. Used when no search provider is
        available. The LLM draws on what it knows from training data —
        which for well-known topics (economics, history, TV shows, science)
        is substantial.

        Args:
            brief: The debate brief (natural language description).

        Returns:
            Structured reference material (markdown string).
        """
        prompt = KNOWLEDGE_COMPILER_PROMPT.format(
            topic=brief,
            perspective_block=(
                f"{_PERSPECTIVE_CONTEXT_TEMPLATE.format(perspective=perspective.strip())}\n"
                if perspective and perspective.strip()
                else ""
            ),
        )

        return self._client.chat(
            messages=[
                {"role": "system", "content": "You are a factual research compiler."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
