"""vox_romana.core.editing — the quality gating pipeline.

This package implements the Editor review loop from the design spec
(agent_instructions.txt sections E-F). It's the production studio's
quality control booth: every chunk of generated content passes through
the Editor before it reaches the final cut.

The pipeline has four moving parts:

    models.py   — Data contracts (OutlineSection, EditorVerdict, ContentResult).
                  Pure data, no behavior. These flow between the other three.

    outline.py  — DirectorOutline: a versioned, mutable show outline that
                  classifies changes as "small" (beats, duration) or "major"
                  (speakers, sections). The Director holds one of these for
                  the entire broadcast.

    editor.py   — ContentEditor: a persistent LLM agent (via AgentSession) that
                  reviews each completed chunk against the outline and a quality
                  rubric (speaker clarity, repetition, narrative coherence).
                  Returns an EditorVerdict (pass/fail with issues).

    retry.py    — RetryLoop: wraps a chunk generation callback with editor
                  gating. On failure, feeds the Editor's notes back to the
                  generator and retries from scratch. Publishes with a warning
                  after max_retries is exhausted.

USAGE:
    from vox_romana.core.editing import (
        ContentEditor, DirectorOutline, OutlineSection, RetryLoop,
    )
    from vox_romana.core.session import AgentSession

    # Build an outline from the Director's plan
    outline = DirectorOutline([
        OutlineSection(id="intro", title="Opening", ...),
        OutlineSection(id="main", title="Core Debate", ...),
    ])

    # Create an editor with a persistent session
    session = AgentSession(client, EDITOR_SYSTEM_PROMPT)
    editor = ContentEditor(session)

    # Wrap generation in a retry loop
    loop = RetryLoop(editor, max_retries=3)
    result = loop.execute(my_generate_fn, "intro", outline)
"""

from .editor import EDITOR_SYSTEM_PROMPT, ContentEditor
from .models import ContentResult, EditorVerdict, OutlineSection, OutlineSnapshot
from .outline import DirectorOutline
from .retry import RetryLoop

__all__ = [
    "EDITOR_SYSTEM_PROMPT",
    "ContentEditor",
    "ContentResult",
    "DirectorOutline",
    "EditorVerdict",
    "OutlineSection",
    "OutlineSnapshot",
    "RetryLoop",
]
