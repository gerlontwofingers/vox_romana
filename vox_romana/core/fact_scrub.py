"""Last-resort rewrite for content that never passes the fact-check gate.

When a conductor exhausts its fact-check retries, publishing the failing
text with a warning leaves known-bad claims in the script. This module asks
the LLM for a targeted edit instead: drop the claims that could not be
verified (and anything that depends on them) while keeping the passage
coherent, continuous, and in the same voice.

The rewrite works on a list of passages (one per event) and must return the
same number of passages, so every event keeps its speaker, pacing, and other
metadata — only ``text`` changes.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from .json_retry import strip_json_fences

logger = logging.getLogger(__name__)

# How much earlier approved content to show the editor for continuity.
_PRIOR_CONTEXT_CHARS = 2000

SCRUB_SYSTEM_PROMPT = """\
You are a meticulous script editor. A fact-checker could not verify certain \
claims in the passages below, even after several rewrite attempts. Your job \
is to edit the passages so those claims are gone while the script still \
reads as if they were never there.

RULES:
- Remove every listed claim, plus any sentence or clause that only makes \
sense because of it. Do NOT replace removed claims with new facts, \
statistics, names, or speculation.
- Keep everything else: same speaker voice, tone, vocabulary, and rhythm. \
Do not touch passages or sentences that are not affected.
- Repair the seams. Remaining text must flow: no dangling pronouns or \
references to removed content, no abrupt topic jumps, no sentences that \
announce something that no longer appears. Add a short connecting phrase \
only when needed to keep continuity, and only using what the passages \
already establish.
- Keep each passage's structural role (an opening still opens, a closing \
still closes) and keep the passages consistent with one another and with the \
EARLIER CONTENT, if provided.
- Stay close to the original length where you can do so without padding in \
new facts. Shorter is acceptable; a non-empty passage must stay non-empty.
- Return exactly one output passage per input passage, in the same order. \
An empty input passage stays an empty string.

Respond with ONLY a JSON array of strings, no commentary.\
"""

_SCRUB_USER_TEMPLATE = """\
{perspective_block}{prior_block}CLAIMS TO REMOVE:
{claims_block}

PASSAGES (JSON array, {count} item(s)):
{passages_json}
"""


def scrub_unresolved_claims(
    client: Any,
    texts: Sequence[str],
    claims: Sequence[Any],
    *,
    prior_context: str | None = None,
    perspective: str | None = None,
) -> list[str] | None:
    """Rewrite *texts* so the *claims* no longer appear.

    Args:
        client:        Anything with ``.chat(messages=[...], **kwargs) -> str``.
        texts:         Passages in script order (one per event).
        claims:        Objects with ``.claim`` and ``.explanation`` (the
                       blocking FactCheckResults from the last failed check).
        prior_context: Earlier approved content, used only for continuity.
        perspective:   Optional show vantage point, so observer-relative
                       phrasing is preserved.

    Returns:
        A list of rewritten passages with the same length as *texts*, or
        ``None`` when the rewrite could not be produced or validated — the
        caller should then fall back to publishing with a warning.
    """
    if client is None or not claims or not any(t and t.strip() for t in texts):
        return None

    claims_block = "\n".join(
        f"- {c.claim} (flagged because: {c.explanation})" for c in claims
    )
    perspective_block = (
        f"SHOW PERSPECTIVE: {perspective.strip()}\n\n"
        if perspective and perspective.strip()
        else ""
    )
    prior_block = ""
    if prior_context and prior_context.strip():
        tail = prior_context.strip()[-_PRIOR_CONTEXT_CHARS:]
        prior_block = (
            "EARLIER CONTENT (for continuity only — do not rewrite or repeat):\n"
            f"{tail}\n\n"
        )

    user_msg = _SCRUB_USER_TEMPLATE.format(
        perspective_block=perspective_block,
        prior_block=prior_block,
        claims_block=claims_block,
        count=len(texts),
        passages_json=json.dumps(list(texts), ensure_ascii=False, indent=2),
    )

    try:
        raw = client.chat(
            messages=[
                {"role": "system", "content": SCRUB_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        data = json.loads(strip_json_fences(raw))
    except Exception as exc:  # noqa: BLE001 — never break generation on a rewrite failure
        logger.warning("fact_scrub: rewrite call failed: %r", exc)
        return None

    if (
        not isinstance(data, list)
        or len(data) != len(texts)
        or not all(isinstance(item, str) for item in data)
    ):
        logger.warning(
            "fact_scrub: rewrite returned an invalid shape (expected %d strings).",
            len(texts),
        )
        return None

    # A non-empty passage must stay non-empty, otherwise the event would
    # render as silence and break pacing.
    for original, rewritten in zip(texts, data):
        if original and original.strip() and not rewritten.strip():
            logger.warning("fact_scrub: rewrite emptied a non-empty passage.")
            return None
    return data
