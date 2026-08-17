"""Retry loop for chunk generation with editor gating.

The retry loop wraps a chunk generation function and subjects each
attempt to editor review. If the editor rejects a chunk, the loop
provides feedback and regenerates from scratch (not patched).

FLOW:
  1. Generate chunk via the provided callback.
  2. Editor reviews the chunk.
  3. If pass → return the chunk.
  4. If fail → feed editor notes back, regenerate from scratch.
  5. Repeat up to max_retries.
  6. If all retries exhausted → return best attempt with warning.

USAGE:
    loop = RetryLoop(editor=editor, max_retries=3)
    result = loop.execute(
        generate_fn=lambda feedback: conductor.generate_chunk(section, feedback),
        content_id="intro",
        outline=director_outline,
    )
    if result.warning:
        print(f"WARNING: {result.warning}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .editor import ContentEditor
from .models import ContentResult, EditorVerdict
from .outline import DirectorOutline

if TYPE_CHECKING:
    from ..conductor import BaseConductor


class RetryLoop:
    """Manages chunk generation with editor gating and configurable retries.

    The generate_fn callback receives editor feedback (empty string on first
    attempt) and returns a tuple of (chunk_content_text, raw_events_list).
    """

    def __init__(
        self,
        editor: ContentEditor,
        *,
        max_retries: int = 3,
        router: "BaseConductor | None" = None,
        target_agents: list[str] | None = None,
    ) -> None:
        self._editor = editor
        self._max_retries = max_retries
        self._router = router
        self._target_agents = target_agents

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def execute(
        self,
        generate_fn: Callable[[str], tuple[str, list[dict]]],
        chunk_id: str,
        outline: DirectorOutline,
    ) -> ContentResult:
        """Execute chunk generation with editor review and retry.

        Args:
            generate_fn: Callable that takes editor feedback (str) and returns
                         (chunk_text, raw_events). Empty feedback on first call.
            chunk_id: Identifier for the chunk.
            outline: Current director outline for context.

        Returns:
            ContentResult with the final content, verdict, and optional warning.
        """
        best_result: ContentResult | None = None
        best_verdict: EditorVerdict | None = None
        feedback = ""

        for attempt in range(1, self._max_retries + 1):
            chunk_text, raw_events = generate_fn(feedback)

            snapshot = outline.snapshot(
                summary=f"Review for {chunk_id} attempt {attempt}"
            )
            verdict = self._editor.review(
                chunk_text, snapshot, content_id=chunk_id
            )

            result = ContentResult(
                content_id=chunk_id,
                content=raw_events,
                outline_version=outline.version,
                attempt_number=attempt,
                editor_verdict=verdict,
            )

            # Track best result (prefer pass, then latest attempt)
            if best_result is None or verdict.status == "pass":
                best_result = result
                best_verdict = verdict

            if verdict.status == "pass":
                return result

            # Build feedback for next attempt
            issues_text = "\n".join(
                f"  - {issue}" for issue in verdict.issues
            )
            recs_text = "\n".join(
                f"  - {rec}" for rec in verdict.recommendations
            )
            feedback = (
                f"EDITOR REJECTION (attempt {attempt}/{self._max_retries}):\n"
                f"{verdict.summary}\n\n"
                f"Issues:\n{issues_text}\n\n"
                f"Recommendations:\n{recs_text}\n\n"
                f"Please regenerate this chunk from scratch addressing "
                f"ALL issues above."
            )
            if self._router is not None:
                self._router._route_editor_rejection(
                    verdict,
                    chunk_id=chunk_id,
                    target_agents=self._target_agents,
                )

        # All retries exhausted — return best attempt with warning
        assert best_result is not None
        assert best_verdict is not None
        unresolved = (
            "; ".join(best_verdict.issues)
            if best_verdict.issues
            else "unspecified"
        )
        best_result.warning = (
            f"Max retries ({self._max_retries}) exhausted for chunk "
            f"'{chunk_id}'. Publishing best available attempt. "
            f"Unresolved issues: {unresolved}"
        )
        return best_result
