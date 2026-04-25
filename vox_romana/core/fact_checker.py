"""FactChecker — verifies generated content against reference material and the web.

The FactChecker is the quality gate for factual accuracy. It takes a block of
generated text (one podcast segment, one debate chunk, etc.), extracts the
concrete factual claims it contains, and checks each one against:

  "quick"    — reference_material only. One extraction call + one verification
               call. Catches direct contradictions with source docs and claims
               that reference material simply doesn't support.

  "thorough" — reference_material PLUS a web search for any claim the reference
               couldn't resolve. Adds ~3-5 search calls and a second verifier
               pass. Catches fabricated statistics, wrong dates, invented names.

HOW IT INTEGRATES:

  The FactChecker is called by the conductor's segment loop (podcast) or the
  ChunkEditor (debates). When has_fabrications is True, the conductor discards
  the segment and regenerates, injecting the fabricated claims as a rejection
  note so the LLM knows exactly what went wrong.

  This is blocking, not advisory. Fabrications prevent publication.
  Only after max_retries are exhausted does the conductor publish with a
  fact_check_warnings field attached to the output so downstream callers know.

TWO-STEP DESIGN:

  Step 1 — Claim Extraction (CLAIM_EXTRACTOR_PROMPT)
    Input:  raw generated text
    Output: JSON list of {claim, reason, search_query} objects

    The extractor focuses only on CONCRETE, SPECIFIC assertions — names, dates,
    statistics, cause-of-death statements, revenue figures, company details.
    Opinions, analysis, and speculative "what if" framings are NOT flagged.

  Step 2a — Quick Verification (CLAIM_VERIFIER_PROMPT)
    Input:  extracted claims + reference_material
    Output: JSON list of {claim, verdict, explanation, source} objects

    Verdicts:
      "verified"     — found and confirmed in reference_material
      "unverifiable" — reference doesn't address this claim (can't confirm or deny)
      "fabricated"   — directly contradicted by reference_material

  Step 2b — Thorough: web search for "unverifiable" claims
    For each "unverifiable" claim: search_provider.search(claim.search_query)
    Second verifier call with search snippets → upgrades to "verified",
    "fabricated", or leaves as "unverifiable" if search is silent.

GRACEFUL DEGRADATION:

  - No reference_material + quick mode + content_mode="factual" → falls back
    to KNOWLEDGE_VERIFIER_PROMPT, which instructs the LLM to check the claims
    against its own training knowledge. This catches clear fabrications even
    when the user hasn't supplied a reference file. "unverifiable" is still
    returned when the LLM is uncertain — only clear contradictions block.

  - No reference_material + content_mode="fictional-universe" → all claims
    come back "unverifiable" (cannot infer fictional canon from LLM general
    knowledge reliably).

  - No search_provider + thorough mode → logs a warning, runs quick behavior.
    Never raises. Never blocks the pipeline unexpectedly.

  - LLM returns unparseable JSON → logs warning, returns empty report (no
    fabrications detected). Conservative fallback — don't block on parse errors.

USAGE:
    from vox_romana.core.client import DeepSeekClient
    from vox_romana.core.search import GoogleSearchProvider
    from vox_romana.core.fact_checker import FactChecker

    checker = FactChecker(
        client=DeepSeekClient(),
        search_provider=GoogleSearchProvider(),
    )

    report = checker.check_chunk(
        chunk_text="Dr. Priya Nair said Pacific AgriLands reported $304,954...",
        chunk_id="seg_002",
        reference_material="Ernie Dosio died April 24, 2026 in a car accident.",
        mode="thorough",
    )

    if report.has_fabrications:
        for result in report.fabricated_claims:
            print(f"FABRICATED: {result.claim}")
            print(f"  → {result.explanation}")
"""

from __future__ import annotations

import json
import logging

from .client import DeepSeekClient
from .config import ContentMode
from .editing.models import ChunkFactCheckReport, FactCheckResult, FlaggedClaim
from .search import BaseSearchProvider, SearchResult, build_fact_check_provider

logger = logging.getLogger(__name__)


# Maximum characters of prior approved content to include in the
# consistency-check prompt. Prevents the prompt from growing unbounded
# as the show gets longer. Trimmed oldest-first by ``_trim_prior_context``.
_MAX_PRIOR_CONTEXT_CHARS = 4000


# ═══════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

CLAIM_EXTRACTOR_PROMPT = """\
You are a fact-extraction specialist. Your job is to identify CONCRETE,
SPECIFIC factual claims in a block of generated text that could be verified
or falsified against external sources.

WHAT TO FLAG — concrete, specific assertions about real things:
  - Names of real people and their biographical details (birth, death, job title)
  - Specific dates or timelines ("died on August 12, 2022", "founded in 2003")
  - Statistics and numerical claims ("$304,954 revenue", "15,000 acre-feet")
  - Company or organisation details (registration, address, employee count)
  - Cause-of-death statements
  - Named studies, legal cases, government programmes, domain names
  - Geographic facts ("no elk in Gabon")

WHAT NOT TO FLAG — do not flag these:
  - Opinions, analysis, speculation, or "I think..." framings
  - Hypothetical or conditional statements ("if X were true...")
  - Qualitative descriptions ("the tone was collegial")
  - Direct questions asked by hosts
  - Claims explicitly marked as uncertain by the speaker ("we don't know...")

OUTPUT FORMAT:
Respond with ONLY a JSON array. Each item:
{{
  "claim": "the verbatim or closely paraphrased claim text",
  "reason": "why this is a checkable factual assertion",
  "search_query": "a short web-search query to verify it (e.g. 'Ernie Dosio death date')"
}}

If there are no checkable factual claims, return an empty array: []
ALWAYS respond with valid JSON only. No other text.
"""

_EXTRACTOR_USER_TEMPLATE = """\
Extract all checkable factual claims from the following generated text:

GENERATED TEXT:
{chunk_text}
"""

CLAIM_VERIFIER_PROMPT = """\
You are a fact-verification specialist. You will be given a list of factual
claims and a block of reference material. Your job is to verify each claim
against the reference material.

VERDICTS:
  "verified"     — The reference material explicitly confirms this claim.
  "unverifiable" — The reference material does not address this claim. You
                   cannot confirm OR deny it from what you've been given.
  "fabricated"   — The reference material directly contradicts this claim.
                   Use this ONLY when there is a clear, direct contradiction —
                   not just absence of confirmation.

CRITICAL RULES:
  - "unverifiable" is NOT the same as "fabricated". Absence of confirmation
    is NOT evidence of fabrication. Only use "fabricated" when the reference
    material explicitly says something different.
  - If reference material is empty or "(none provided)", ALL claims must be
    "unverifiable". You cannot fabricate a "fabricated" verdict.
  - Your explanation must be specific: quote the relevant part of the
    reference material, or note exactly what is absent.

OUTPUT FORMAT:
Respond with ONLY a JSON array. Each item:
{{
  "claim": "the original claim text",
  "verdict": "verified" | "unverifiable" | "fabricated",
  "explanation": "specific explanation citing the reference material",
  "source": "reference_material" | "none"
}}

ALWAYS respond with valid JSON only. No other text.
"""

_VERIFIER_USER_TEMPLATE = """\
REFERENCE MATERIAL:
{reference_material}

CLAIMS TO VERIFY:
{claims_json}
"""


# ── LLM knowledge fallback verifier ───────────────────────────────────────
# Used when content_mode == "factual" and no reference_material is provided.
# Instead of the reference-material verifier (which correctly returns
# "unverifiable" for everything when the reference is empty), this prompt
# instructs the LLM to reason from its training knowledge.
#
# Conservative by design: the LLM should only say "fabricated" when it has
# high confidence the claim is wrong (wrong death date for a known figure,
# invented revenue for a public company, etc.). For contested, uncertain, or
# post-training-cutoff facts it must return "unverifiable", not "fabricated".

KNOWLEDGE_VERIFIER_PROMPT = """\
You are a fact-verification specialist. You will be given a list of factual
claims made in generated audio content. No reference document has been
provided, so you must verify each claim using your own training knowledge.

VERDICTS:
  "verified"     — You are confident this claim is correct based on your
                   training knowledge.
  "unverifiable" — You are uncertain, the claim is outside your training
                   data, or the claim involves recent/contested facts you
                   cannot reliably judge. When in doubt, use this.
  "fabricated"   — You are HIGHLY CONFIDENT this claim is factually wrong
                   based on well-established knowledge (e.g. wrong age or
                   birth year for a person you know, invented statistics
                   for a known entity, obvious geographic error, wrong
                   death date for someone whose death was reported).

CRITICAL RULES:
  - USE YOUR KNOWLEDGE. If you have specific, confidently-known facts about
    a person or entity — their birth year, age, role, death date — apply
    that knowledge and return "verified" or "fabricated" as appropriate.
    Do NOT suppress correct knowledge just because the subject is not a
    global celebrity.
  - Only use "unverifiable" when the entity is genuinely unfamiliar to you,
    or the specific claim is a detail you don't actually have in your
    training data (e.g. a private company's internal revenue figure, a
    claim about an event after your training cutoff, a highly contested
    fact with no clear consensus).
  - Do NOT use "fabricated" for claims you merely can't confirm. Absence of
    knowledge is "unverifiable", not "fabricated".
  - Your explanation must state specifically what your training knowledge
    says, or clearly explain why you are genuinely uncertain.

OUTPUT FORMAT:
Respond with ONLY a JSON array. Each item:
{{
  "claim": "the original claim text",
  "verdict": "verified" | "unverifiable" | "fabricated",
  "explanation": "specific explanation citing your knowledge or uncertainty",
  "source": "llm_knowledge"
}}

ALWAYS respond with valid JSON only. No other text.
"""

_KNOWLEDGE_VERIFIER_USER_TEMPLATE = """\
CLAIMS TO VERIFY (use your own training knowledge as the truth source):
{claims_json}
"""


SEARCH_VERIFIER_PROMPT = """\
You are a fact-verification specialist. You will be given a list of claims
that could not be verified from reference material, along with web search
results. Your job is to verify each claim against the search results.

VERDICTS:
  "verified"     — The search results explicitly confirm this claim.
  "unverifiable" — The search results do not address this claim clearly enough
                   to confirm or deny it.
  "fabricated"   — The search results directly contradict this claim.

CRITICAL RULES:
  - Only use "fabricated" when a search result clearly and directly contradicts
    the claim. Hedging language in snippets is not sufficient.
  - If search results are empty or off-topic, verdict must be "unverifiable".
  - Your explanation must cite the specific search result (title or snippet)
    that supports your verdict.

OUTPUT FORMAT:
Respond with ONLY a JSON array. Each item:
{{
  "claim": "the original claim text",
  "verdict": "verified" | "unverifiable" | "fabricated",
  "explanation": "specific explanation citing the search result",
  "source": "web search: <result title or URL>"
}}

ALWAYS respond with valid JSON only. No other text.
"""

_SEARCH_VERIFIER_USER_TEMPLATE = """\
CLAIMS TO VERIFY (these could not be resolved from reference material):
{claims_json}

WEB SEARCH RESULTS:
{search_results}
"""


# ── Fictional-universe canon verifier ──────────────────────────────────────
# Used when content_mode == "fictional-universe". The truth standard is
# established CANON of the named fictional world (Star Wars, Marvel, etc.),
# NOT real-world facts. Luke Skywalker is from Tatooine, not Kansas.
#
# Verdicts mirror the factual verifier so the downstream report shape stays
# the same. The ChunkEditor decides whether to BLOCK on "fabricated" (factual
# mode) or only WARN (fictional-universe mode).

CANON_VERIFIER_PROMPT = """\
You are a canon-verification specialist. You will be given a list of claims
made in a piece of fiction set in an established fictional universe (Star
Wars, Marvel, Lord of the Rings, Harry Potter, Star Trek, etc.) and a block
of reference material describing the canon.

Your job is to flag claims that DIRECTLY CONTRADICT established canon.

VERDICTS:
  "verified"     — The reference material confirms this claim matches canon.
  "unverifiable" — The reference material does not address this claim.
                   You cannot confirm or deny it from what you have.
                   ABSENCE of confirmation is NOT a violation.
  "fabricated"   — The claim CLEARLY contradicts established canon
                   (e.g. "Luke Skywalker is from Kansas" when canon places
                   him on Tatooine). Use sparingly: only when the
                   contradiction is unambiguous.

CRITICAL RULES:
  - Storytelling often INVENTS new details that fit within canon
    (a new minor character, a new location on a known planet). These are
    "unverifiable", NOT "fabricated". Only flag direct contradictions of
    KNOWN canon facts.
  - If reference material is empty or "(none provided)", ALL claims are
    "unverifiable". You cannot fabricate a "fabricated" verdict.
  - Your explanation must cite the specific canon detail that is contradicted.

OUTPUT FORMAT:
Respond with ONLY a JSON array. Each item:
{{
  "claim": "the original claim text",
  "verdict": "verified" | "unverifiable" | "fabricated",
  "explanation": "specific explanation citing the canon detail",
  "source": "reference_material" | "none"
}}

ALWAYS respond with valid JSON only. No other text.
"""


# ── Intra-show consistency verifier ────────────────────────────────────────
# Used after the fabrication/canon check, when prior_context is provided.
# Catches a different class of error: the new segment/turn/chapter
# contradicts something an EARLIER segment/turn/chapter in the same show
# already asserted (e.g. earlier said "the trial began in 2017", later
# says "the trial started in 2019").

CONSISTENCY_VERIFIER_PROMPT = """\
You are an intra-show consistency checker. You will be given:

  1. PRIOR CONTENT — text from earlier segments/turns of the SAME show
     that has already been approved.
  2. NEW CLAIMS — concrete factual assertions in the latest segment.

Your job is to flag any NEW CLAIM that DIRECTLY CONTRADICTS something
asserted in the PRIOR CONTENT.

VERDICTS:
  "verified"          — The new claim is consistent with prior content
                        (or simply not addressed by it).
  "contradicts_prior" — The new claim directly conflicts with a fact
                        previously stated. (e.g. earlier said "X was 32",
                        new says "X was 47".)

CRITICAL RULES:
  - ELABORATION is fine: adding NEW detail not previously mentioned is
    NOT a contradiction. Only flag direct conflicts.
  - PARAPHRASE is fine: restating an earlier fact in different words is
    consistent.
  - DEBATE / DIALOGUE: if PRIOR CONTENT shows different speakers
    asserting different opinions, that is NOT a contradiction \u2014 it is
    debate. Only flag a contradiction when the SAME SPEAKER asserts
    incompatible facts, OR when both statements are presented as
    objective narrative truth.
  - CHARACTER GROWTH / CHANGES OVER TIME: if context makes clear that
    something CHANGED ("years later, X moved to..."), that is not a
    contradiction.
  - When in doubt, prefer "verified". Only flag clear, unambiguous
    contradictions.

OUTPUT FORMAT:
Respond with ONLY a JSON array. One item per NEW CLAIM:
{{
  "claim": "the original claim text",
  "verdict": "verified" | "contradicts_prior",
  "explanation": "if contradicts_prior, quote the conflicting earlier statement",
  "source": "prior_context" | "none"
}}

ALWAYS respond with valid JSON only. No other text.
"""

_CONSISTENCY_USER_TEMPLATE = """\
PRIOR CONTENT (earlier segments/turns of this same show, already approved):
{prior_context}

NEW CLAIMS (from the latest segment, to check for contradictions):
{claims_json}
"""


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# Max search results to fetch per unverifiable claim (thorough mode).
_RESULTS_PER_CLAIM = 3

# Max number of unverifiable claims to search (cost guard — avoid runaway
# spending on segments with many unchecked claims).
_MAX_SEARCH_CLAIMS = 8


# ═══════════════════════════════════════════════════════════════════════════
# FACT CHECKER
# ═══════════════════════════════════════════════════════════════════════════

class FactChecker:
    """Verifies factual claims in generated content.

    One FactChecker instance can be reused across many chunks in a single
    run. It is stateless between calls — no session, no history.

    Attributes:
        _client:          DeepSeekClient for LLM calls.
        _search_provider: Optional search backend for thorough mode.
    """

    def __init__(
        self,
        client: DeepSeekClient,
        search_provider: BaseSearchProvider | None = None,
    ) -> None:
        """Initialise the FactChecker.

        Args:
            client:          DeepSeekClient for LLM calls.
            search_provider: Any BaseSearchProvider for thorough mode.
                             If None and mode="thorough", falls back to quick.
        """
        self._client = client
        self._search = search_provider

    # ── Public API ──────────────────────────────────────────────────────

    def check_chunk(
        self,
        chunk_text: str,
        *,
        chunk_id: str = "",
        reference_material: str | None = None,
        mode: str = "quick",
        content_mode: ContentMode = "factual",
        prior_context: str | None = None,
    ) -> ChunkFactCheckReport:
        """Check a generated chunk for fabricated factual claims.

        Args:
            chunk_text:         The full generated text to check.
            chunk_id:           Identifier for logging and the report.
            reference_material: Factual source text to verify against.
                                If None, all claims come back "unverifiable".
            mode:               "quick" \u2014 reference_material only.
                                "thorough" \u2014 reference + web search for
                                             unresolved claims.
            content_mode:       "factual" (default) verifies against the real
                                world. "fictional-universe" verifies against
                                established canon (Star Wars, etc.). "pure-fiction"
                                skips fact-checking entirely (zero LLM calls).
            prior_context:      Optional concatenation of earlier approved
                                segments/turns/chapters from the SAME show.
                                When provided (and content_mode is not
                                "pure-fiction"), an additional consistency
                                pass flags claims that contradict prior content.
                                Use :func:`_trim_prior_context` to keep the
                                prompt size bounded.

        Returns:
            ChunkFactCheckReport. Check ``report.has_fabrications`` and
            ``report.has_contradictions`` to decide whether to block.
        """
        # Pure fiction: no real-world or canon claims to check, no prior-show
        # consistency contract to enforce. Skip entirely \u2014 zero LLM calls.
        if content_mode == "pure-fiction":
            return ChunkFactCheckReport(
                chunk_id=chunk_id,
                mode=mode,
                summary=f"[{chunk_id}] content_mode='pure-fiction' \u2014 fact-check skipped.",
            )

        if not chunk_text.strip():
            return ChunkFactCheckReport(chunk_id=chunk_id, mode=mode, summary="Empty chunk \u2014 nothing to check.")

        # \u2500\u2500 Step 1: Extract checkable claims \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        flagged = self.extract_claims(chunk_text)
        if not flagged:
            return ChunkFactCheckReport(
                chunk_id=chunk_id,
                mode=mode,
                results=[],
                summary="No checkable factual claims found in this chunk.",
            )

        # \u2500\u2500 Step 2a: Verify against reference material / canon \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        results = self.verify_claims(
            flagged,
            reference_material=reference_material,
            content_mode=content_mode,
        )

        # \u2500\u2500 Step 2b: Thorough \u2014 search for unverifiable claims \u2500\u2500\u2500\u2500\u2500\u2500
        if mode == "thorough":
            if self._search is None:
                logger.warning(
                    "FactChecker: mode='thorough' requested but no search_provider "
                    "is configured. Falling back to quick (LLM knowledge) check — "
                    "Google web search upgrade skipped."
                )
            else:
                results = self._upgrade_with_search(results, flagged)

        # \u2500\u2500 Step 3: Intra-show consistency pass \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        # Only runs when prior_context is provided. Operates on the SAME
        # extracted claims \u2014 no second extraction call needed.
        if prior_context and prior_context.strip():
            consistency_results = self._check_consistency(flagged, prior_context)
            results = self._merge_consistency_results(results, consistency_results)

        # \u2500\u2500 Build the report \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        summary = self._build_summary(results, chunk_id)
        return ChunkFactCheckReport(
            chunk_id=chunk_id,
            mode=mode,
            results=results,
            summary=summary,
        )

    # ── Extraction ──────────────────────────────────────────────────────

    def extract_claims(self, chunk_text: str) -> list[FlaggedClaim]:
        """Extract checkable factual claims from generated text.

        One LLM call. Returns an empty list on parse errors (conservative:
        don't block on extractor failures).

        Args:
            chunk_text: The generated text to extract claims from.

        Returns:
            List of FlaggedClaim objects.
        """
        user_msg = _EXTRACTOR_USER_TEMPLATE.format(chunk_text=chunk_text)
        try:
            raw = self._client.chat(
                messages=[
                    {"role": "system", "content": CLAIM_EXTRACTOR_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
            )
            return self.parse_claims(raw)
        except Exception as exc:
            logger.warning("FactChecker.extract_claims failed: %r", exc)
            return []

    def parse_claims(self, raw_json: str) -> list[FlaggedClaim]:
        """Parse the extractor LLM response into FlaggedClaim objects.

        Separated for testability — pass canned JSON, get deterministic output.

        Args:
            raw_json: Raw LLM response (should be a JSON array).

        Returns:
            List of FlaggedClaim objects. Empty list on parse errors.
        """
        cleaned = _strip_fences(raw_json)
        try:
            items = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "FactChecker: claim extractor returned unparseable JSON (%r). "
                "First 200 chars: %r",
                exc, raw_json[:200],
            )
            return []

        if not isinstance(items, list):
            logger.warning(
                "FactChecker: claim extractor returned %r instead of a list.",
                type(items),
            )
            return []

        claims: list[FlaggedClaim] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("claim", "")).strip()
            reason = str(item.get("reason", "")).strip()
            search_query = item.get("search_query") or None
            if text:
                claims.append(FlaggedClaim(
                    text=text,
                    reason=reason,
                    search_query=str(search_query) if search_query else None,
                ))
        return claims

    # ── Verification ────────────────────────────────────────────────────

    def verify_claims(
        self,
        claims: list[FlaggedClaim],
        *,
        reference_material: str | None = None,
        content_mode: ContentMode = "factual",
    ) -> list[FactCheckResult]:
        """Verify claims against reference material (quick mode core step).

        One LLM call. Returns "unverifiable" for all claims on parse errors.

        Args:
            claims:             List of FlaggedClaim objects to verify.
            reference_material: Source text to check against. If None or
                                empty, all claims return "unverifiable".
            content_mode:       "factual" uses CLAIM_VERIFIER_PROMPT (real-world
                                truth); "fictional-universe" uses
                                CANON_VERIFIER_PROMPT (canon truth). Anything
                                else defaults to factual.

        Returns:
            List of FactCheckResult objects (one per claim).
        """
        if not claims:
            return []

        claims_json = json.dumps(
            [{"claim": c.text, "search_query": c.search_query} for c in claims],
            indent=2,
        )

        no_ref = not reference_material or not reference_material.strip()

        # ── Choose prompt + user message based on mode and reference ──────
        if no_ref and content_mode == "factual":
            # No reference supplied: fall back to LLM training knowledge.
            # This allows the verifier to catch clear fabrications (wrong death
            # dates, invented statistics) even without a user-provided document.
            system_prompt = KNOWLEDGE_VERIFIER_PROMPT
            user_msg = _KNOWLEDGE_VERIFIER_USER_TEMPLATE.format(claims_json=claims_json)
        else:
            ref_block = reference_material.strip() if reference_material else "(none provided)"
            user_msg = _VERIFIER_USER_TEMPLATE.format(
                reference_material=ref_block,
                claims_json=claims_json,
            )
            system_prompt = (
                CANON_VERIFIER_PROMPT
                if content_mode == "fictional-universe"
                else CLAIM_VERIFIER_PROMPT
            )

        try:
            raw = self._client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
            )
            return self.parse_results(raw, fallback_claims=claims)
        except Exception as exc:
            logger.warning("FactChecker.verify_claims failed: %r", exc)
            # Conservative fallback: unverifiable, not fabricated.
            return [
                FactCheckResult(
                    claim=c.text,
                    verdict="unverifiable",
                    explanation=f"Verification call failed: {exc!r}",
                    source="none",
                )
                for c in claims
            ]

    def parse_results(
        self,
        raw_json: str,
        *,
        fallback_claims: list[FlaggedClaim] | None = None,
    ) -> list[FactCheckResult]:
        """Parse the verifier LLM response into FactCheckResult objects.

        Separated for testability.

        Args:
            raw_json:        Raw LLM response (should be a JSON array).
            fallback_claims: If provided, used to build "unverifiable"
                             fallbacks when JSON can't be parsed.

        Returns:
            List of FactCheckResult objects.
        """
        cleaned = _strip_fences(raw_json)
        try:
            items = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "FactChecker: verifier returned unparseable JSON (%r). "
                "First 200 chars: %r",
                exc, raw_json[:200],
            )
            if fallback_claims:
                return [
                    FactCheckResult(
                        claim=c.text,
                        verdict="unverifiable",
                        explanation="Verifier response could not be parsed.",
                        source="none",
                    )
                    for c in fallback_claims
                ]
            return []

        if not isinstance(items, list):
            logger.warning(
                "FactChecker: verifier returned %r instead of a list.",
                type(items),
            )
            return []

        results: list[FactCheckResult] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            raw_verdict = str(item.get("verdict", "unverifiable")).strip().lower()
            # Normalise to valid verdicts only.
            if raw_verdict not in ("verified", "unverifiable", "fabricated", "contradicts_prior"):
                raw_verdict = "unverifiable"
            explanation = str(item.get("explanation", "")).strip()
            source = item.get("source") or "none"
            if claim:
                results.append(FactCheckResult(
                    claim=claim,
                    verdict=raw_verdict,  # type: ignore[arg-type]
                    explanation=explanation,
                    source=str(source),
                ))
        return results

    # ── Thorough: web-search upgrade ────────────────────────────────────

    def _upgrade_with_search(
        self,
        results: list[FactCheckResult],
        flagged: list[FlaggedClaim],
    ) -> list[FactCheckResult]:
        """Upgrade 'unverifiable' results using web search (thorough mode).

        For each "unverifiable" result, searches the web using the
        corresponding FlaggedClaim.search_query. Feeds results back into
        a second verifier call to upgrade verdicts where possible.

        Leaves "verified" and "fabricated" results untouched.

        Args:
            results: Results from the reference-material verification pass.
            flagged: Original FlaggedClaim list (for search_query lookup).

        Returns:
            Updated list of FactCheckResult objects.
        """
        # Build a lookup: claim text → FlaggedClaim (for search_query).
        claim_to_flagged: dict[str, FlaggedClaim] = {c.text: c for c in flagged}

        # Collect unverifiable results that have a search query.
        to_search: list[FactCheckResult] = []
        for r in results:
            if r.verdict == "unverifiable":
                fc = claim_to_flagged.get(r.claim)
                if fc and fc.search_query:
                    to_search.append(r)

        # Cap to avoid runaway search costs.
        to_search = to_search[:_MAX_SEARCH_CLAIMS]

        if not to_search:
            return results

        # Search the web for each unverifiable claim.
        search_context: list[str] = []
        for result in to_search:
            fc = claim_to_flagged[result.claim]
            query = fc.search_query or result.claim
            try:
                hits: list[SearchResult] = self._search.search(  # type: ignore[union-attr]
                    query, num_results=_RESULTS_PER_CLAIM
                )
            except Exception as exc:
                logger.warning(
                    "FactChecker: search failed for query %r: %r", query, exc
                )
                continue

            if not hits:
                continue

            # Format hits for the LLM.
            snippets = "\n".join(
                f"TITLE: {h.title}\nURL: {h.url}\nSNIPPET: {h.snippet}"
                for h in hits
                if h.snippet.strip()
            )
            if snippets:
                search_context.append(
                    f"--- Search results for: {query!r} ---\n{snippets}"
                )

        if not search_context:
            return results

        # Second verifier pass with search results.
        claims_json = json.dumps(
            [{"claim": r.claim} for r in to_search], indent=2
        )
        search_results_block = "\n\n".join(search_context)
        user_msg = _SEARCH_VERIFIER_USER_TEMPLATE.format(
            claims_json=claims_json,
            search_results=search_results_block,
        )

        try:
            raw = self._client.chat(
                messages=[
                    {"role": "system", "content": SEARCH_VERIFIER_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
            )
            upgraded = self.parse_results(raw)
        except Exception as exc:
            logger.warning("FactChecker: search verifier call failed: %r", exc)
            return results

        # Merge: replace the "unverifiable" entries that were searched.
        upgraded_by_claim: dict[str, FactCheckResult] = {
            r.claim: r for r in upgraded
        }
        merged: list[FactCheckResult] = []
        for r in results:
            if r.verdict == "unverifiable" and r.claim in upgraded_by_claim:
                merged.append(upgraded_by_claim[r.claim])
            else:
                merged.append(r)
        return merged

    # \u2500\u2500 Intra-show consistency check \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _check_consistency(
        self,
        flagged: list[FlaggedClaim],
        prior_context: str,
    ) -> list[FactCheckResult]:
        """Check each extracted claim against prior approved show content.

        One LLM call. Returns an empty list on parse errors (conservative:
        do not block on consistency-checker failures).

        Args:
            flagged:        Claims extracted from the current chunk.
            prior_context:  Concatenated text of earlier approved
                            segments/turns/chapters in this same show.

        Returns:
            List of FactCheckResult objects with verdict either
            "verified" or "contradicts_prior".
        """
        if not flagged:
            return []

        trimmed = _trim_prior_context(prior_context)
        claims_json = json.dumps(
            [{"claim": c.text} for c in flagged],
            indent=2,
        )
        user_msg = _CONSISTENCY_USER_TEMPLATE.format(
            prior_context=trimmed,
            claims_json=claims_json,
        )

        try:
            raw = self._client.chat(
                messages=[
                    {"role": "system", "content": CONSISTENCY_VERIFIER_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
            )
        except Exception as exc:
            logger.warning("FactChecker._check_consistency LLM call failed: %r", exc)
            return []

        return self.parse_results(raw)

    @staticmethod
    def _merge_consistency_results(
        primary: list[FactCheckResult],
        consistency: list[FactCheckResult],
    ) -> list[FactCheckResult]:
        """Merge consistency results into the primary results list.

        For each consistency result with verdict "contradicts_prior", we
        either:
          - replace the matching primary entry whose verdict is NOT
            "fabricated" (don't overwrite a stronger fabricated verdict), OR
          - append it as an additional result if no matching claim is found.

        "verified" verdicts from the consistency pass are dropped \u2014 the
        primary pass already speaks to verification, and we don't want
        spurious noise.
        """
        if not consistency:
            return primary

        primary_by_claim: dict[str, int] = {
            r.claim: i for i, r in enumerate(primary)
        }
        merged = list(primary)

        for c in consistency:
            if c.verdict != "contradicts_prior":
                continue
            idx = primary_by_claim.get(c.claim)
            if idx is None:
                merged.append(c)
                continue
            existing = merged[idx]
            # Don't downgrade a "fabricated" verdict to "contradicts_prior".
            # Both are blocking in factual mode; leave the stronger signal.
            if existing.verdict == "fabricated":
                continue
            merged[idx] = c

        return merged

    # \u2500\u2500 Default-construction helper \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    @classmethod
    def build_default(cls, client: DeepSeekClient) -> FactChecker:
        """Construct a FactChecker with the standard search-provider fallback.

        Mirrors the ``_enable_research`` ladder used in ``run_show.py``:
        try Google, silently fall back to LLM-only when credentials are
        missing. This is the convenience constructor every conductor uses
        unless it needs special wiring.
        """
        return cls(
            client=client,
            search_provider=build_fact_check_provider(),
        )

    # \u2500\u2500 Summary builder \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    @staticmethod
    def _build_summary(results: list[FactCheckResult], chunk_id: str) -> str:
        """Build a human-readable summary of the fact-check results.

        The summary is injected into the conductor's rejection note so
        the regeneration LLM call knows exactly what to fix.
        """
        if not results:
            return f"[{chunk_id}] No checkable claims found."

        fabricated = [r for r in results if r.verdict == "fabricated"]
        contradicted = [r for r in results if r.verdict == "contradicts_prior"]
        verified = [r for r in results if r.verdict == "verified"]
        unverifiable = [r for r in results if r.verdict == "unverifiable"]

        parts: list[str] = [
            f"[{chunk_id}] Checked {len(results)} claim(s): "
            f"{len(verified)} verified, "
            f"{len(unverifiable)} unverifiable, "
            f"{len(fabricated)} FABRICATED, "
            f"{len(contradicted)} CONTRADICT PRIOR."
        ]

        if fabricated:
            parts.append("FABRICATED CLAIMS (must be corrected before publishing):")
            for r in fabricated:
                parts.append(f"  \u2022 {r.claim}")
                parts.append(f"    Reason: {r.explanation}")

        if contradicted:
            parts.append(
                "CONTRADICTIONS WITH PRIOR APPROVED CONTENT "
                "(must be reconciled before publishing):"
            )
            for r in contradicted:
                parts.append(f"  \u2022 {r.claim}")
                parts.append(f"    Conflict: {r.explanation}")

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _strip_fences(raw: str) -> str:
    """Strip markdown code fences from an LLM response.

    Thin compatibility wrapper around :func:`json_retry.strip_json_fences`.
    Kept as a private name so existing call sites in this module need no
    changes; new code should import ``strip_json_fences`` directly.
    """
    from .json_retry import strip_json_fences
    return strip_json_fences(raw)


def _trim_prior_context(
    text: str,
    max_chars: int = _MAX_PRIOR_CONTEXT_CHARS,
) -> str:
    """Trim prior-context text to ``max_chars``, keeping the most recent content.

    Drops OLDEST content first \u2014 the most recent segments/turns/chapters are
    the most relevant for catching contradictions in the new chunk. The
    returned text is prefixed with an "[\u2026 earlier content trimmed \u2026]"
    marker so the LLM knows the context is a tail slice.

    Args:
        text:      Raw concatenated prior-context text (any size).
        max_chars: Maximum character length of the returned string.

    Returns:
        The trimmed text. If ``text`` is already \u2264 ``max_chars``,
        returned unchanged.
    """
    if len(text) <= max_chars:
        return text
    marker = "[\u2026 earlier content trimmed \u2026]\n"
    keep = max_chars - len(marker)
    if keep <= 0:
        return text[-max_chars:]
    return marker + text[-keep:]
