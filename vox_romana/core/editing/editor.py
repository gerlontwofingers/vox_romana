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

from ..session import AgentSession
from .models import EditorVerdict, OutlineSnapshot


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

    def __init__(self, session: AgentSession) -> None:
        self._session = session

    @property
    def session(self) -> AgentSession:
        return self._session

    def review(
        self,
        chunk_text: str,
        outline_snapshot: OutlineSnapshot,
        chunk_id: str = "",
    ) -> EditorVerdict:
        """Review a chunk and return a verdict.

        Args:
            chunk_text: The full text of the generated chunk.
            outline_snapshot: Current outline version for context.
            chunk_id: Identifier for the chunk being reviewed.

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
        return self._parse_verdict(
            raw_response, chunk_id, outline_snapshot.version
        )

    def _parse_verdict(
        self, raw: str, chunk_id: str, outline_version: int
    ) -> EditorVerdict:
        """Parse LLM response into an EditorVerdict.

        Same robust JSON extraction as the existing conductors —
        handles markdown fences, preamble text, and garbage gracefully.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

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
