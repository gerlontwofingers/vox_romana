"""Loader for the ``llm:`` block in ``vox_romana.yaml``.

Thin shim around :func:`vox_romana.core.paths._load_config` that returns
just the ``llm`` sub-dict (or ``{}`` when absent).  Kept as a separate
module so callers don't have to know the YAML loading internals and so
the lens rack has a single ``from_config`` entry-point.

YAML SHAPE
──────────

.. code-block:: yaml

    llm:
      # Default lens for everything not overridden below.
      default: cloud-deepseek

      # Per-role / per-character overrides.
      roles:
        translator: local-qwen-polyglot
        "character:Salvatore": local-llama-dim

      # Optional: register additional Ollama models as named lenses.
      lenses:
        - name: local-mistral-small
          backend: local
          model: mistral-small:latest
          personality_hint: "Terse French-leaning generalist."
          language_strengths: [en, fr]
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_llm_config() -> dict[str, Any]:
    """Return the ``llm:`` block from ``vox_romana.yaml``.

    Returns an empty dict when the file or block is missing — callers
    can pass the result straight to
    :meth:`vox_romana.core.lens.LLMLensRack.from_config`.
    """
    try:
        from vox_romana.core.paths import _load_config  # type: ignore[attr-defined]
        return _load_config().get("llm") or {}
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("Failed to load llm: block from vox_romana.yaml: %s", exc)
        return {}


def parse_lens_overrides(specs: list[str] | None) -> dict[str, str]:
    """Parse ``--llm-lens role=lens_name`` argv values.

    Args:
        specs: Raw values collected by ``argparse`` for the
               ``--llm-lens`` flag (may be ``None``).

    Returns:
        Mapping of ``role`` → ``lens_name``.

    Raises:
        ValueError: When any spec is malformed.
    """
    out: dict[str, str] = {}
    for raw in specs or []:
        if "=" not in raw:
            raise ValueError(
                f"Bad --llm-lens value '{raw}': expected 'role=lens_name'."
            )
        role, lens_name = raw.split("=", 1)
        role = role.strip()
        lens_name = lens_name.strip()
        if not role or not lens_name:
            raise ValueError(
                f"Bad --llm-lens value '{raw}': empty role or lens name."
            )
        out[role] = lens_name
    return out
