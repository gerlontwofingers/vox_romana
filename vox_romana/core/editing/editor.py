"""The Editor — non-interactive quality gating agent.

The Editor reviews each completed chunk and decides pass or fail.
It uses a persistent AgentSession for the duration of the broadcast
and can optionally consult the fact-checker.

The Editor's review lens:
  1. Speaker clarity — Can the audience tell who is speaking?
  2. Repetition — Has this been said before?
  3. Intention fidelity — Does the chunk serve the prompt and outline?
  4. Instruction compliance — Are instructions being followed?
  5. Factual accuracy — Optionally consult the fact-checker.
  6. Narrative quality — Is this engaging?

USAGE:
    session = AgentSession(client, EDITOR_SYSTEM_PROMPT)
    editor = ChunkEditor(session)
    verdict = editor.review(chunk_text, outline_snapshot)
"""

from __future__ import annotations

import json
import logging

from ..config import ContentMode
from ..fact_checker import FactChecker
from ..session import AgentSession
from .models import EditorVerdict, OutlineSnapshot

logger = logging.getLogger(__name__)


EDITOR_SYSTEM_PROMPT = """\
You are the Editor for a broadcast production system. Your job is to review \
each chunk of generated content and decide whether it passes quality standards.

You are the AUDIENCE'S ADVOCATE. Think about what a listener hears:
- Can they tell who is speaking at all times? (Speaker clarity)
- Have they heard this before in this broadcast? (Repetition)
- Does this serve the stated purpose of the show? (Intention fidelity)
- Are all instructions from the prompt being followed? (Instruction compliance)
- Is the content factually sound? (Factual accuracy — flag doubts)
- Is this engaging and does it move the show forward? (Narrative quality)

You must respond with a JSON object:
{
  "status": "pass" or "fail",
  "summary": "One paragraph explaining your judgment.",
  "issues": ["list of specific problems, if any"],
  "recommendations": ["list of specific fixes for each issue"]
}

Be honest and specific. If a chunk is mediocre but passable, pass it with \
notes. Only fail chunks that would genuinely hurt the broadcast quality. \
When you fail a chunk, your recommendations should be actionable enough \
that a regeneration attempt can address every issue.

ALWAYS respond with valid JSON only. No text outside the JSON block.\
"""

_REVIEW_PROMPT = """\
CHUNK TO REVIEW (chunk_id: {chunk_id}):

OUTLINE CONTEXT (v{outline_version}):
{outline_text}

CHUNK CONTENT:
{chunk_text}

Review this chunk against the outline and quality standards. Respond with JSON.\
"""


class ChunkEditor:
    """Non-interactive editor that reviews chunks via a persistent LLM session.

    One ChunkEditor per broadcast. The session accumulates context so later
    reviews are informed by earlier ones.
    """

    def __init__(
        self,
        session: AgentSession,
        fact_checker: FactChecker | None = None,
        reference_material: str | None = None,
        content_mode: ContentMode = "factual",
    ) -> None:
        self._session = session
        self._fact_checker = fact_checker
        self._reference_material = reference_material
        self._content_mode = content_mode

    @property
    def session(self) -> AgentSession:
        return self._session

    def review(
        self,
        chunk_text: str,
        outline_snapshot: OutlineSnapshot,
        chunk_id: str = "",
        prior_context: str | None = None,
    ) -> EditorVerdict:
        """Review a chunk and return a verdict.

        Args:
            chunk_text:       The full text of the generated chunk.
            outline_snapshot: Current outline version for context.
            chunk_id:         Identifier for the chunk being reviewed.
            prior_context:    Optional concatenated text of earlier approved
                              segments/turns/chapters in this same show.
                              Passed to the FactChecker for intra-show
                              consistency checking.

        Returns:
            EditorVerdict with status, issues, and recommendations.
        """
        prompt = _REVIEW_PROMPT.format(
            chunk_id=chunk_id,
            outline_version=outline_snapshot.version,
            outline_text=outline_snapshot.to_broadcast_text(),
            chunk_text=chunk_text,
        )

        raw_response = self._session.send(prompt)
        verdict = self._parse_verdict(
            raw_response, chunk_id, outline_snapshot.version
        )

        # ── Fact-check hook ──────────────────────────────────────────
        # Blocking behaviour depends on content_mode:
        #   factual             — fabricated AND contradicts_prior both BLOCK
        #   fictional-universe  — fabricated / contradicts_prior become WARNINGS
        #                          (CANON_WARN / CANON_CONTRADICTION labels),
        #                          verdict.status is never flipped to "fail"
        #   pure-fiction        — FactChecker returns empty report; nothing to do
        if self._fact_checker is not None:
            try:
                report = self._fact_checker.check_chunk(
                    chunk_text,
                    chunk_id=chunk_id,
                    reference_material=self._reference_material,
                    mode="quick",
                    content_mode=self._content_mode,
                    prior_context=prior_context,
                )

                if self._content_mode == "fictional-universe":
                    # Non-blocking: annotate but do not fail.
                    for fc_result in report.fabricated_claims:
                        verdict.issues.append(
                            f"CANON_WARN: {fc_result.claim}"
                        )
                        verdict.recommendations.append(
                            f"Canon note — {fc_result.claim}: {fc_result.explanation}"
                        )
                    for fc_result in report.contradicted_claims:
                        verdict.issues.append(
                            f"CANON_CONTRADICTION: {fc_result.claim}"
                        )
                        verdict.recommendations.append(
                            f"Intra-show canon note — {fc_result.claim}: {fc_result.explanation}"
                        )
                    if report.fabricated_claims or report.contradicted_claims:
                        logger.info(
                            "ChunkEditor: chunk %r has %d canon warn(s) and "
                            "%d intra-show canon contradiction(s) — non-blocking "
                            "(fictional-universe mode).",
                            chunk_id,
                            len(report.fabricated_claims),
                            len(report.contradicted_claims),
                        )
                else:
                    # factual (or any unknown mode): block on both.
                    blocking_claims = (
                        [("FABRICATED", r) for r in report.fabricated_claims]
                        + [("CONTRADICTION", r) for r in report.contradicted_claims]
                    )
                    if blocking_claims:
                        verdict.status = "fail"
                        for label, fc_result in blocking_claims:
                            verdict.issues.append(
                                f"{label}: {fc_result.claim}"
                            )
                            verdict.recommendations.append(
                                f"Remove or correct — "
                                f"{fc_result.claim}: {fc_result.explanation}"
                            )
                        logger.warning(
                            "ChunkEditor: chunk %r failed fact-check — "
                            "%d fabricated, %d contradicting prior.",
                            chunk_id,
                            len(report.fabricated_claims),
                            len(report.contradicted_claims),
                        )
            except Exception as exc:
                logger.warning(
                    "ChunkEditor: fact-check raised unexpectedly for chunk %r: %r. "
                    "Continuing without blocking.",
                    chunk_id, exc,
                )

        return verdict

    def _parse_verdict(
        self, raw: str, chunk_id: str, outline_version: int
    ) -> EditorVerdict:
        """Parse LLM response into an EditorVerdict.

        Same robust JSON extraction as the existing conductors —
        handles markdown fences, preamble text, and garbage gracefully.
        """
        from ..json_retry import strip_json_fences
        cleaned = strip_json_fences(raw)

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

        try:
            data = json.loads(cleaned)
            return EditorVerdict(
                status=data.get("status", "pass"),
                chunk_id=chunk_id,
                outline_version=outline_version,
                summary=data.get("summary", ""),
                issues=data.get("issues", []),
                recommendations=data.get("recommendations", []),
            )
        except (json.JSONDecodeError, TypeError):
            # Can't parse → conservative: pass with a warning
            return EditorVerdict(
                status="pass",
                chunk_id=chunk_id,
                outline_version=outline_version,
                summary=(
                    f"Editor response could not be parsed. "
                    f"Raw: {raw[:200]}"
                ),
                issues=["editor_parse_failure"],
                recommendations=["Manual review recommended"],
            )
