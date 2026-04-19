"""vox_romana.shows — content-type implementations.

Each subpackage is a self-contained "studio" for one show type:

    debates/  — Multi-participant debate generation (Oxford, crossfire, etc.).
    news/     — News broadcast generation (anchors, segments, rundown).

Each studio follows the same lifecycle pattern:
    bootstrap.py  → Reads a brief, produces a typed config.
    conductor.py  → Runs the multi-agent turn loop, produces events.
    serializer.py → Saves the output as JSON.
    config.py     → Show-specific config dataclass.
    models.py     → Show-specific event/output dataclasses.
    prompts.py    — System prompts for the show's agents.

The unified entry points (run_show.py, render_show.py) route to the
correct studio automatically. You can also import directly:
    from vox_romana.shows.debates import DebateConductor
    from vox_romana.shows.news import NewsConductor
"""