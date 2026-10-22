"""Crew member identity, episodic memory, and cross-run consolidation.

Crew members are first-class entities outside any particular show run.
A director, for example, can work on news, audiobooks, and debates and
accumulate wisdom (and a stable identity) across all of them.

File layout on disk::

    <crew_dir>/
      <crew_id>/
        profile.yaml      # stable identity — never auto-modified
        short_term.jsonl  # recent run summaries, salience-scored
        long_term.jsonl   # semantic generalizations (consolidated from short-term)

Three-layer memory model
------------------------
Layer 1 — Profile:
    Stable identity. Loaded at ``_build_agents()`` time and prepended to
    the crew member's role system prompt. Human-editable, never written
    by the system.

Layer 2 — Semantic / long-term:
    Interpreted cross-run generalizations produced by an LLM consolidation
    call.  Show-tagged so a director whose profile spans news and debates
    can hold show-specific wisdom separately.

Layer 3 — Episodic / short-term:
    One entry per run.  Salience-scored so smooth runs fade (low salience)
    while difficult, high-drama, or error-prone runs survive longer.
    When the entry count exceeds ``CONSOLIDATION_THRESHOLD``, a
    consolidation pass runs automatically and prunes low-salience entries.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CREW_DIR: Path = Path.home() / ".vox_romana" / "crew"

# When short_term.jsonl reaches this many entries, consolidation runs.
CONSOLIDATION_THRESHOLD: int = 20

# Number of entries to keep in short_term after a consolidation pass.
# We retain the top-5 by salience AND the 5 most recent, de-duplicated.
KEEP_HIGH_SALIENCE: int = 5
KEEP_RECENT: int = 5

# Cap on how many long-term generalizations are injected into a system
# prompt. Long-term memory grows without bound on disk; the prompt budget
# does not. Most-recent entries win.
MAX_LONG_TERM_IN_PROMPT: int = 8


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Profile
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CrewProfile:
    """Stable identity for a crew member.  Never auto-modified by the system.

    Fields mirror the YAML schema in ``profile.yaml``.  All text fields
    are optional; absent fields are simply omitted from the injected block.

    Attributes:
        name:                    Display name of the crew member.
        role:                    Functional role: "director" or "factchecker".
        aesthetic_philosophy:    The crew member's overarching creative values.
        stylistic_tendencies:    Habitual style choices (pacing, rhythm, tone).
        pacing_approach:         How they think about runtime and dramatic tempo.
        conflict_stance:         (directors) How they handle on-air disagreements.
        threshold_calibration:   (factcheckers) How conservatively they flag claims.
    """

    name: str
    role: str  # "director" | "factchecker"
    aesthetic_philosophy: str = ""
    stylistic_tendencies: str = ""
    pacing_approach: str = ""
    conflict_stance: str = ""
    threshold_calibration: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "CrewProfile":
        """Load a CrewProfile from a YAML file.

        Missing keys are silently defaulted.  The profile name defaults to
        the parent directory name when not specified.

        Args:
            path: Absolute path to ``profile.yaml``.

        Raises:
            OSError:     When the file cannot be read.
            yaml.YAMLError: When the file is malformed YAML.
        """
        import yaml  # pyyaml — listed in requirements.txt

        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(
            name=str(data.get("name", path.parent.name)),
            role=str(data.get("role", "director")),
            aesthetic_philosophy=str(data.get("aesthetic_philosophy", "")),
            stylistic_tendencies=str(data.get("stylistic_tendencies", "")),
            pacing_approach=str(data.get("pacing_approach", "")),
            conflict_stance=str(data.get("conflict_stance", "")),
            threshold_calibration=str(data.get("threshold_calibration", "")),
        )

    def format_for_prompt(self) -> str:
        """Return a compact text block for injection into an LLM system prompt.

        Only non-empty fields are included so the injected block stays brief.
        The returned string is suitable for prepending to any role prompt,
        separated by a blank line.
        """
        lines = [f"CREW IDENTITY: {self.name}"]
        for label, value in (
            ("Aesthetic philosophy", self.aesthetic_philosophy),
            ("Stylistic tendencies", self.stylistic_tendencies),
            ("Pacing approach", self.pacing_approach),
            ("Conflict stance", self.conflict_stance),
            ("Threshold calibration", self.threshold_calibration),
        ):
            if value:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Crew member handle
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CrewMember:
    """A crew member assigned to a run.

    Attributes:
        crew_id:    Directory name under the crew root (e.g. ``marcus_webb``).
        role:       Functional role: ``"director"`` or ``"factchecker"``.
        profile:    Loaded profile, or ``None`` when no ``profile.yaml`` found.
        long_term:  Layer-2 semantic generalizations loaded from
                    ``long_term.jsonl``. Empty until enough runs have
                    accumulated for a consolidation pass to fire.
    """

    crew_id: str
    role: str
    profile: CrewProfile | None = None
    long_term: list[dict[str, Any]] = field(default_factory=list)

    def profile_text(self) -> str:
        """Return the formatted profile block, or '' when no profile is set."""
        if self.profile is None:
            return ""
        return self.profile.format_for_prompt()

    def memory_text(
        self,
        *,
        show_type: str | None = None,
        limit: int = MAX_LONG_TERM_IN_PROMPT,
    ) -> str:
        """Return the formatted long-term memory block for prompt injection.

        Args:
            show_type: When given, only generalizations tagged for this show
                       type (or untagged, i.e. universally applicable) are
                       included.
            limit:     Maximum number of generalizations to include.
        """
        return format_long_term_for_prompt(
            self.long_term, show_type=show_type, limit=limit,
        )


def format_long_term_for_prompt(
    records: list[dict[str, Any]],
    *,
    show_type: str | None = None,
    limit: int = MAX_LONG_TERM_IN_PROMPT,
) -> str:
    """Format Layer-2 generalizations into a compact system-prompt block.

    Records whose ``show_tags`` is empty are treated as universally
    applicable and always kept. Records tagged for other show types are
    dropped when *show_type* is supplied. The most recent *limit* records
    survive (long_term.jsonl is append-ordered, so later = newer).

    Returns an empty string when nothing is applicable, so callers can
    concatenate unconditionally.
    """
    if not records:
        return ""

    applicable: list[str] = []
    for rec in records:
        text = str(rec.get("text", "")).strip()
        if not text:
            continue
        tags = rec.get("show_tags") or []
        if show_type and tags and show_type not in tags:
            continue
        applicable.append(text)

    if not applicable:
        return ""

    lines = ["WHAT YOU HAVE LEARNED FROM PAST PRODUCTIONS:"]
    lines.extend(f"- {text}" for text in applicable[-limit:])
    return "\n".join(lines)


def load_long_term(
    crew_id: str,
    *,
    crew_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Read all Layer-2 generalizations from ``long_term.jsonl``.

    Returns an empty list when the file is absent (the normal state until
    a crew member has accumulated enough runs to trigger consolidation).
    Malformed lines are skipped with a warning rather than raising.
    """
    base = Path(crew_dir or DEFAULT_CREW_DIR)
    path = base / crew_id / "long_term.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("crew: skipping malformed long_term line: %s", exc)
    return records


def load_crew_member(
    crew_id: str,
    role: str,
    *,
    crew_dir: Path | None = None,
) -> CrewMember:
    """Build a :class:`CrewMember` by reading ``profile.yaml`` from disk.

    Also loads Layer-2 long-term memory from ``long_term.jsonl`` when
    present, so accumulated cross-run wisdom can be injected alongside the
    stable profile.

    Returns a member with ``profile=None`` (and logs a warning) when the
    file is absent or malformed — so callers don't need to check for errors.

    Args:
        crew_id:  Directory name under the crew root.
        role:     Functional role string (``"director"`` or ``"factchecker"``).
        crew_dir: Base directory for crew data.  Defaults to
                  ``~/.vox_romana/crew/``.
    """
    base = Path(crew_dir or DEFAULT_CREW_DIR)
    profile_path = base / crew_id / "profile.yaml"
    profile: CrewProfile | None = None
    if profile_path.exists():
        try:
            profile = CrewProfile.from_yaml(profile_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "crew: failed to load profile for %r from %s: %s",
                crew_id, profile_path, exc,
            )
    else:
        logger.debug(
            "crew: no profile.yaml for %r at %s — running without identity.",
            crew_id, profile_path,
        )
    return CrewMember(
        crew_id=crew_id,
        role=role,
        profile=profile,
        long_term=load_long_term(crew_id, crew_dir=crew_dir),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Episodic / short-term memory
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EpisodicEntry:
    """One run's summary, written to ``short_term.jsonl`` at the end of a run.

    Attributes:
        run_id:               Unique identifier for this run (e.g. UUID or timestamp).
        show_type:            Content type: ``"debate"``, ``"podcast"``, etc.
        title:                Episode title from the generated config.
        date:                 ISO-8601 UTC timestamp of the run.
        salience_score:       0–10 float from :func:`compute_salience`.
        summary:              One-paragraph plain-English description of the run.
        production_log_stats: Lightweight stats derived from the production log
                              (issue count, challenge count, etc.).
    """

    run_id: str
    show_type: str
    title: str
    date: str
    salience_score: float
    summary: str
    production_log_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─── Salience scoring ──────────────────────────────────────────────────────


def compute_salience(production_log: list[dict[str, Any]]) -> float:
    """Score a run's salience from its structured production log.

    Higher salience = more notable; smooth runs score near zero.
    Score caps at **10.0**.

    Rubric (additive):

    +1.0  per ``issue_report`` event (cast flagged a problem)
    +2.0  per ``agent_message`` whose channel is ``"directive"`` and whose
          payload reason is ``"retries_exhausted"`` (fact-check loop gave up)
    +1.0  per ``agent_message`` broadcast with phase ``"director_note"``
          (director changed course mid-run — unusual)
    +1.0  per ``agent_message`` challenge where the payload records the claim
          as still fabricated after the loop (unresolved fabrication)

    Args:
        production_log: The ``ConductorResult.production_log`` list.

    Returns:
        A float in [0.0, 10.0].
    """
    score = 0.0
    for entry in production_log:
        event_type = entry.get("event_type", "")
        if event_type == "issue_report":
            score += 1.0
        elif event_type == "agent_message":
            channel = entry.get("channel", "")
            payload = entry.get("payload") or {}
            if channel == "directive":
                if payload.get("reason") == "retries_exhausted":
                    score += 2.0
            elif channel == "broadcast":
                if payload.get("phase") == "director_note":
                    score += 1.0
            elif channel == "challenge":
                # A fabricated claim that survived the challenge loop
                if (entry.get("result") or {}).get("verdict") == "fabricated":
                    score += 1.0
    return min(score, 10.0)


# ─── Write + read ──────────────────────────────────────────────────────────


def write_episodic_entry(
    crew_id: str,
    entry: EpisodicEntry,
    *,
    crew_dir: Path | None = None,
) -> None:
    """Append *entry* to ``<crew_dir>/<crew_id>/short_term.jsonl``.

    Creates the directory tree if it does not exist.  Never overwrites
    existing entries — always appends.

    Args:
        crew_id:  Directory name under the crew root.
        entry:    Populated :class:`EpisodicEntry` to persist.
        crew_dir: Base directory for crew data.
    """
    base = Path(crew_dir or DEFAULT_CREW_DIR)
    member_dir = base / crew_id
    member_dir.mkdir(parents=True, exist_ok=True)
    path = member_dir / "short_term.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.to_dict(), ensure_ascii=True, sort_keys=True))
        fh.write("\n")
    logger.debug(
        "crew: wrote episodic entry for %r (salience=%.1f) to %s",
        crew_id, entry.salience_score, path,
    )


def _load_short_term(path: Path) -> list[dict[str, Any]]:
    """Load all valid JSONL lines from a short_term file."""
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("crew: skipping malformed short_term line: %s", exc)
    return entries


def _prune_short_term(
    entries: list[dict[str, Any]],
    *,
    keep_high_salience: int = KEEP_HIGH_SALIENCE,
    keep_recent: int = KEEP_RECENT,
) -> list[dict[str, Any]]:
    """Return the subset of entries to keep after consolidation.

    Retains the ``keep_high_salience`` entries with the highest
    ``salience_score`` and the ``keep_recent`` most recent entries (by
    list position, which corresponds to write order).  Duplicates are
    removed and the result is sorted chronologically by date.
    """
    if not entries:
        return []
    # Top-K by salience
    by_salience = sorted(
        range(len(entries)),
        key=lambda i: float(entries[i].get("salience_score", 0.0)),
        reverse=True,
    )
    keep_indices: set[int] = set(by_salience[:keep_high_salience])
    # Most recent K (by file position — later = more recent)
    for i in range(max(0, len(entries) - keep_recent), len(entries)):
        keep_indices.add(i)
    kept = [entries[i] for i in sorted(keep_indices)]
    return kept


def _strip_json_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence line (```json or ```)
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        # Drop the closing fence
        if text.rstrip().endswith("```"):
            text = text[: text.rstrip().rfind("```")]
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Semantic / long-term consolidation
# ─────────────────────────────────────────────────────────────────────────────


def consolidate_if_needed(
    crew_id: str,
    *,
    client: Any,
    crew_dir: Path | None = None,
    threshold: int = CONSOLIDATION_THRESHOLD,
) -> bool:
    """Consolidate short-term memory into long-term when the entry count hits threshold.

    Reads ``short_term.jsonl``.  If the entry count is below *threshold*,
    returns ``False`` immediately.  Otherwise:

    1. Calls the LLM to produce 3–6 actionable generalizations.
    2. Appends them to ``long_term.jsonl``.
    3. Prunes ``short_term.jsonl`` to the top-salience + most-recent entries.

    The LLM receives only the first 8 000 characters of the serialised
    short-term log so the prompt stays within a reasonable token budget.

    Args:
        crew_id:   Directory name under the crew root.
        client:    A :class:`DeepSeekClient` (or any object with a
                   ``.chat(messages=...)`` method).
        crew_dir:  Base directory for crew data.
        threshold: Minimum entry count before consolidation fires.

    Returns:
        ``True`` when consolidation ran, ``False`` otherwise.
    """
    base = Path(crew_dir or DEFAULT_CREW_DIR)
    short_path = base / crew_id / "short_term.jsonl"
    long_path = base / crew_id / "long_term.jsonl"

    entries = _load_short_term(short_path)
    if len(entries) < threshold:
        return False

    logger.info(
        "crew: consolidating %d short-term entries for %r (threshold=%d).",
        len(entries), crew_id, threshold,
    )

    # Build the prompt — cap payload to stay within a sane token budget.
    entries_json = json.dumps(entries, indent=2, ensure_ascii=True)
    if len(entries_json) > 8_000:
        entries_json = entries_json[:8_000] + "\n... (truncated)"

    prompt = (
        "You are a production debriefer who distils experience into craft wisdom.\n\n"
        "Below is a crew member's recent production log. Each entry describes one "
        "show run and carries a salience_score (0–10; higher = more notable).\n\n"
        "Your task: produce 3–6 concise, actionable generalizations the crew member "
        "has learned from these runs. Each generalization must be:\n"
        "- Specific and actionable ('When X, do Y')\n"
        "- Tagged with the show type(s) it applies to\n"
        "- Weighted toward high-salience entries\n"
        "- Written in first person ('I have found that...')\n\n"
        "Respond with valid JSON only:\n"
        '{"generalizations": ['
        '{"text": "...", "show_tags": ["debate", "news"], "derived_from_runs": 3}'
        "]}\n\n"
        f"SHORT-TERM LOG:\n{entries_json}"
    )

    try:
        raw = client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a production debriefer. Respond with valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        data = json.loads(_strip_json_fences(raw))
        generalizations: list[dict[str, Any]] = data.get("generalizations", [])
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "crew: consolidation LLM call failed for %r: %s", crew_id, exc
        )
        return False

    if not generalizations:
        logger.warning(
            "crew: consolidation returned no generalizations for %r — skipping.",
            crew_id,
        )
        return False

    # Append generalizations to long_term.jsonl
    long_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_at = datetime.now(timezone.utc).isoformat()
    with long_path.open("a", encoding="utf-8") as fh:
        for gen in generalizations:
            record: dict[str, Any] = {
                "consolidated_at": consolidated_at,
                "crew_id": crew_id,
                "source_run_count": len(entries),
                **gen,
            }
            fh.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            fh.write("\n")

    # Prune short_term
    kept = _prune_short_term(entries)
    with short_path.open("w", encoding="utf-8") as fh:
        for entry in kept:
            fh.write(json.dumps(entry, ensure_ascii=True, sort_keys=True))
            fh.write("\n")

    logger.info(
        "crew: %r — consolidated %d entries → %d generalizations; "
        "short_term pruned to %d entries.",
        crew_id, len(entries), len(generalizations), len(kept),
    )
    return True
