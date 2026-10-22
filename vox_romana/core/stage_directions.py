"""Shared stage-direction detection.

Canonical convention (see docs in each show's prompts.py): scene actions
that end up inline in an event's ``text`` field must use ONLY [square
brackets]. Parentheses are reserved for prose asides (acronym expansions,
numeric asides) and are never treated as a scene action.

This module is the single source of truth for "does this parenthetical
look like a stage direction" so the TTS renderer (which strips them) and
the conductor-side format validator (which flags convention violations)
can't drift out of sync.
"""

from __future__ import annotations

import re

# Matches any (parenthetical) block in utterance text, e.g.:
#   (laughs)  (dramatic pause)  (Leaning in, voice low)
STAGE_DIR_RE = re.compile(r'\(([^)]*)\)')

# Matches square-bracket stage directions, e.g.:
#   [Dr. Waffleford holds the pizza aloft.]
#   [The audience gasps.]
BRACKET_STAGE_DIR_RE = re.compile(r'\[[^\]]*\]')


def is_stage_direction(content: str) -> bool:
    """Return True if a parenthetical's contents look like a performance cue.

    Stage directions contain no digits and no acronyms (two or more
    consecutive uppercase letters). Numeric asides and acronym expansions
    are treated as prose and left untouched.  Long prose explanations with
    conjunctions and verb phrases are also left untouched.
    """
    stripped = content.strip()
    # Numeric asides: "(3.5 million)", "(down 12%)"
    if re.search(r'\d', stripped):
        return False
    # Acronym expansions: "(UBI)", "(GDP growth)"
    if re.search(r'[A-Z]{2,}', stripped):
        return False
    # Long prose parentheticals with conjunctions/relative clauses are not cues
    if len(stripped.split()) > 15:
        return False
    return True


def find_parenthetical_stage_directions(text: str) -> list[str]:
    """Return every ``(parenthetical)`` fragment in *text* that reads as a
    stage direction — i.e. a canonical-format violation (should have been
    written as ``[square brackets]`` instead).
    """
    if not text:
        return []
    return [
        m.group(1).strip()
        for m in STAGE_DIR_RE.finditer(text)
        if is_stage_direction(m.group(1))
    ]
