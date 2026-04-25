"""JSON validation and retry for structural LLM calls.

Many pipeline steps (architect, director brief, profile builder) require
the LLM to return valid JSON with specific keys. LLMs sometimes return
malformed JSON, wrap it in markdown fences, or miss required fields.

This module provides:

    parse_json_response()  — Extract and parse JSON from an LLM response.
                             Handles markdown fences, preamble text, etc.

    json_retry()           — Call a generation function, validate the JSON
                             response, and retry with feedback if needed.

USAGE:
    from vox_romana.core.json_retry import json_retry, parse_json_response

    # Simple parsing:
    data = parse_json_response(raw_llm_output)

    # With retry:
    result = json_retry(
        generate_fn=lambda feedback: client.chat(messages + feedback_msg(feedback)),
        required_keys=["title", "segments"],
        max_retries=2,
    )
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class JsonParseError(Exception):
    """Raised when JSON cannot be extracted from an LLM response."""

    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def strip_json_fences(raw: str) -> str:
    """Strip surrounding markdown code fences from an LLM response.

    LLMs often wrap JSON output in ```json ... ``` (or just ``` ... ```) even
    when explicitly told not to. This helper drops the opening fence (with
    optional language tag) and the closing fence, returning the inner text
    stripped of leading/trailing whitespace. Input without fences is returned
    unchanged (apart from a top-level strip()).

    This is the canonical fence-stripping helper for the codebase. Other
    modules (fact_checker, editing.editor) should import this rather than
    re-implement it.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Drop the opening fence (and optional language tag) and closing fence.
        cleaned = "\n".join(lines[1:-1]).strip()
    return cleaned


def parse_json_response(raw: str) -> dict:
    """Extract and parse a JSON object from an LLM response.

    Handles common LLM output patterns:
    - Clean JSON
    - JSON wrapped in ```json ... ``` markdown fences
    - JSON with preamble/postamble text
    - Nested JSON (finds outermost { ... })

    Args:
        raw: The raw LLM response string.

    Returns:
        The parsed dict.

    Raises:
        JsonParseError: If no valid JSON object can be extracted.
    """
    cleaned = strip_json_fences(raw)

    # Try direct parse first (most common case).
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # Find the outermost JSON object via brace matching.
    start = cleaned.find("{")
    if start == -1:
        raise JsonParseError("No JSON object found in response", raw)

    # Walk forward to find the matching closing brace.
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict):
                        return data
                except (json.JSONDecodeError, TypeError):
                    break

    # Last resort: find last } and try start..end.
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    raise JsonParseError(
        f"Could not parse JSON from response (length={len(raw)})", raw
    )


def json_retry(
    generate_fn,
    *,
    required_keys: list[str] | None = None,
    max_retries: int = 2,
) -> dict:
    """Call a generation function, validate JSON, retry on failure.

    Args:
        generate_fn:   Callable that takes a feedback string (empty on first
                       call) and returns the raw LLM response string.
        required_keys: If provided, the parsed JSON must contain all these
                       top-level keys. Missing keys trigger a retry.
        max_retries:   How many additional attempts after the first failure.
                       Total attempts = 1 + max_retries.

    Returns:
        The validated parsed dict.

    Raises:
        JsonParseError: If all attempts fail to produce valid JSON.
    """
    feedback = ""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 2):  # +2 because range is exclusive
        raw = generate_fn(feedback)

        try:
            data = parse_json_response(raw)
        except JsonParseError as e:
            last_error = e
            feedback = (
                f"Your previous response (attempt {attempt}) was not valid JSON. "
                f"Please respond with ONLY a JSON object — no markdown fences, "
                f"no explanation text. Just the raw JSON."
            )
            logger.warning(
                "JSON parse failed on attempt %d: %s", attempt, str(e)
            )
            continue

        # Check required keys.
        if required_keys:
            missing = [k for k in required_keys if k not in data]
            if missing:
                feedback = (
                    f"Your JSON response (attempt {attempt}) is missing required "
                    f"keys: {', '.join(missing)}. Please include ALL of these "
                    f"top-level keys: {', '.join(required_keys)}."
                )
                logger.warning(
                    "JSON missing keys on attempt %d: %s", attempt, missing
                )
                last_error = JsonParseError(
                    f"Missing keys: {missing}", raw
                )
                continue

        return data

    # All attempts failed.
    raise last_error or JsonParseError("All retry attempts failed", "")
