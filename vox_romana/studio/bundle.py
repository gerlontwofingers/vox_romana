"""Studio bundle — materialize a studio record into a portable directory.

Shot 5.2-c of the Studio Instantiation feature.  This module turns a
:class:`~vox_romana.core.catalog.SeriesRecord` (the Catalog's source of
truth) into a self-contained directory of files a human can read, version,
or hand to another system:

    studio/{slug}/
    ├── studio.yaml        # series metadata (title, show_type, tone, slug)
    ├── bible.md           # the writers' bible, as Markdown
    ├── audience.json      # AudienceProfile serialization
    ├── prompts/
    │   ├── episode_1_brief.txt   # optional (only when a draft exists)
    │   └── template.txt          # reusable episode-brief template
    └── auditions/
        └── queue.json     # empty manifest — real audition wiring is deferred

The bundle is a *materialized export snapshot*, NOT an alternative store —
the Catalog remains authoritative.  ``materialize_bundle`` works on any
``SeriesRecord``, so ``--mode series export`` can snapshot a pre-existing
series (not just a freshly instantiated one); ``zip_bundle`` packages the
directory into a single archive for sharing.

PyYAML is used for ``studio.yaml`` (it's already a project dependency).
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path

from ..core.catalog import SeriesRecord
from ..planning.episode_planner import EpisodeBrief


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE_TXT = """\
# Episode brief template — {title}

Fill this in and pass it to the pipeline:

    python run.py prompts/episode_N.txt --series {series_id}

Show type: {show_type}
Tone:      {tone}
"""


def _studio_yaml_payload(series: SeriesRecord, slug: str) -> dict:
    """Build the dict serialized into ``studio.yaml``."""
    payload = {
        "series_id": series.series_id,
        "slug": slug,
        "title": series.title,
        "show_type": series.show_type,
        "tone": series.tone,
        "created_at": series.created_at,
    }
    if series.audience_profile is not None:
        payload["audience"] = asdict(series.audience_profile)
    else:
        payload["audience"] = None
    return payload


def _empty_audition_queue(series_id: str) -> str:
    """Serialize the deferred audition queue manifest."""
    return json.dumps(
        {
            "series_id": series_id,
            "note": (
                "Audition queue manifest. Real audition scheduling is "
                "deferred; approve assets via `--mode audition` and "
                "`--mode series approve`."
            ),
            "queue": [],
        },
        indent=2,
        ensure_ascii=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Materialization
# ─────────────────────────────────────────────────────────────────────────────


def materialize_bundle(
    series: SeriesRecord,
    dest_root: str | Path,
    *,
    slug: str | None = None,
    episode_brief: EpisodeBrief | None = None,
) -> Path:
    """Write a portable studio bundle directory and return its path.

    Args:
        series:        The authoritative SeriesRecord (from Catalog).
        dest_root:     Parent directory under which ``studio/{slug}/``
                       is created.
        slug:          Directory name for the bundle.  Defaults to the
                       series UUID (unique, always safe).  Freshly
                       instantiated studios pass the human-friendly slug
                       from Stage-1 meta instead.
        episode_brief: Optional EpisodeBrief — written as
                       ``prompts/episode_1_brief.txt`` when present.

    Returns:
        The resolved bundle directory path.
    """
    dir_slug = slug or series.series_id
    bundle_dir = Path(dest_root) / "studio" / dir_slug

    # Re-create the bundle directory fresh each time (idempotent snapshot).
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    (bundle_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "auditions").mkdir(parents=True, exist_ok=True)

    # studio.yaml
    import yaml  # pyyaml — listed in requirements.txt

    payload = _studio_yaml_payload(series, dir_slug)
    (bundle_dir / "studio.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # bible.md
    (bundle_dir / "bible.md").write_text(
        series.bible or "(empty bible)", encoding="utf-8",
    )

    # audience.json
    audience_json = (
        series.audience_profile.to_json()
        if series.audience_profile is not None
        else "{}"
    )
    (bundle_dir / "audience.json").write_text(audience_json, encoding="utf-8")

    # prompts/
    (bundle_dir / "prompts" / "template.txt").write_text(
        _TEMPLATE_TXT.format(
            title=series.title,
            series_id=series.series_id,
            show_type=series.show_type or "(any)",
            tone=series.tone or "(no fixed tone)",
        ),
        encoding="utf-8",
    )
    if episode_brief is not None and episode_brief.prompt_text:
        (bundle_dir / "prompts" / "episode_1_brief.txt").write_text(
            episode_brief.prompt_text.rstrip() + "\n", encoding="utf-8",
        )

    # auditions/queue.json (empty manifest)
    (bundle_dir / "auditions" / "queue.json").write_text(
        _empty_audition_queue(series.series_id), encoding="utf-8",
    )

    return bundle_dir


# ─────────────────────────────────────────────────────────────────────────────
# Zip export
# ─────────────────────────────────────────────────────────────────────────────


def zip_bundle(bundle_dir: str | Path) -> Path:
    """Zip an existing bundle directory into ``<bundle_dir>.zip``.

    Args:
        bundle_dir: Path to a bundle produced by
                    :func:`materialize_bundle`.

    Returns:
        The resolved archive path.

    Raises:
        FileNotFoundError: If ``bundle_dir`` does not exist.
    """
    src = Path(bundle_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {src}")

    archive = src.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(src.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(src))
    return archive
