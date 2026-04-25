"""Auto-classify a brief into one of three :class:`ContentMode` values.

This module exists so the FactChecker knows WHAT KIND of truth to enforce
on a given show:

    factual              \u2014 real-world claims must match reality
    fictional-universe   \u2014 claims must match established canon (Star Wars, etc.)
    pure-fiction         \u2014 wholly original; nothing to fact-check

THE WHY
=======
A user writes a brief like::

    "An audiobook chapter about Luke Skywalker's childhood on Tatooine."

Without classification we have no way to tell the FactChecker that
"Luke is from Kansas" is wrong \u2014 it is not a real-world fact, but it IS
a canon violation. Conversely, "Hannibal crossed the Alps with elephants
in 218 BC" is a real-world claim and any fabrication is unacceptable.

By inferring a :class:`ContentMode` once at bootstrap and then threading
it through every FactChecker call, downstream prompts can be specialised
to the right kind of truth.


CONSERVATIVE FALLBACK
=====================
On any error (LLM failure, unparseable answer, etc.) we return "factual".
The user has stated explicitly that the system must NEVER produce content
that is intended to be factual but is not. The reverse error \u2014 fact-checking
a wholly fictional show \u2014 is annoying but never produces falsehoods, so
"factual" is the safe default.
"""

from __future__ import annotations

import logging
from typing import get_args

from .client import DeepSeekClient
from .config import ContentMode

logger = logging.getLogger(__name__)


_VALID_MODES: tuple[ContentMode, ...] = get_args(ContentMode)




CONTENT_MODE_PROMPT = """\
You are a content classifier. Read the brief below and decide which of these
three modes the resulting show belongs to.

MODES (answer with EXACTLY ONE of these three words, lowercase, nothing else):

  factual
      Real-world content. Discusses real people, real events, real places,
      real research. The audience expects everything stated as fact to be
      true in reality.

  fictional-universe
      Content set in an established fictional world (Star Wars, Marvel, Lord
      of the Rings, Harry Potter, Star Trek, Doctor Who, the Witcher, etc.).
      The audience expects details to match established canon \u2014 e.g. Luke
      Skywalker is from Tatooine, not Kansas.

  pure-fiction
      Wholly original fiction with no real-world claims and no pre-existing
      canon. Original short stories, made-up worlds, fictional case studies.

If the brief is ambiguous or mixes modes, prefer 'factual'. Better to
over-check than to let a false claim through.

Output format: one word. No punctuation. No explanation.

BRIEF:
"""


class ContentModeClassifier:
    """Classify a brief into a :class:`ContentMode`.

    Usage::

        classifier = ContentModeClassifier(client)
        mode = classifier.classify(brief="...", show_type="podcast")

    The classifier is stateless; you can reuse one instance across many
    calls or instantiate a fresh one each time \u2014 same effect.
    """

    def __init__(self, client: DeepSeekClient) -> None:
        self._client = client

    def classify(self, brief: str, show_type: str) -> ContentMode:
        """Return the inferred :class:`ContentMode` for ``brief``.

        Args:
            brief:      The user's natural-language description of the show.
            show_type:  Short identifier of the show kind. Recognised values
                        include "news", "debate", "podcast", "audiobook".

        Returns:
            One of "factual", "fictional-universe", or "pure-fiction".
            On any error or ambiguity the result is "factual".
        """
        if not brief or not brief.strip():
            logger.debug("ContentModeClassifier: empty brief, defaulting to 'factual'")
            return "factual"

        prompt = CONTENT_MODE_PROMPT + brief.strip()
        try:
            raw = self._client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0,
            )
        except Exception:  # noqa: BLE001 \u2014 conservative fallback on ANY LLM error
            logger.warning(
                "ContentModeClassifier: LLM call failed, defaulting to 'factual'",
                exc_info=True,
            )
            return "factual"

        return _parse_mode(raw)


def _parse_mode(raw: str) -> ContentMode:
    """Best-effort extraction of one of the three modes from an LLM reply.

    The prompt asks for a single word, but real LLMs sometimes wrap the
    answer in punctuation, quotes, or a brief preamble. We scan the reply
    for the first matching mode token and accept it. If nothing matches,
    we fall back to "factual".
    """
    if not raw:
        return "factual"

    # Lowercase + strip surrounding whitespace/punctuation/quotes.
    cleaned = raw.strip().strip("\"'`.,!?\n ").lower()

    # Exact match first \u2014 the well-behaved case.
    if cleaned in _VALID_MODES:
        return cleaned  # type: ignore[return-value]

    # Otherwise scan the entire reply for the first occurrence of any mode
    # token. This handles "The mode is: fictional-universe." style replies.
    lowered = raw.lower()
    earliest_idx = -1
    earliest_mode: ContentMode = "factual"
    for mode in _VALID_MODES:
        idx = lowered.find(mode)
        if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
            earliest_idx = idx
            earliest_mode = mode

    if earliest_idx == -1:
        logger.warning(
            "ContentModeClassifier: could not parse mode from %r; "
            "defaulting to 'factual'",
            raw[:200],
        )
        return "factual"

    return earliest_mode


__all__ = ["ContentModeClassifier", "CONTENT_MODE_PROMPT"]
