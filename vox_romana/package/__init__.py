"""YouTube packaging — generate SRT captions and show notes from a rendered episode.

Usage (standalone):

    >>> from vox_romana.package import package_episode, PackageResult
    >>> result = package_episode(
    ...     json_path=Path("output/text/debates/ubi.json"),
    ...     event_durations={0: 3.2, 1: 4.7, ...},   # from RenderResult
    ...     generate_notes=True,
    ...     client=DeepSeekClient(),
    ... )
    >>> print(result.srt_path, result.notes_path)

Or just the building blocks:

    >>> from vox_romana.package import make_srt, extract_chapters, generate_show_notes

CLI surface:

    python run.py --mode package --script-input output/text/debates/ubi.json
    python run.py prompts/ubi.txt --mode video --package
    python run.py prompts/ubi.txt --package                    # add-on for full mode

Files produced (default: next to the show JSON):
    {stem}.srt                — SRT captions for YouTube Studio
    {stem}_show_notes.txt     — YouTube description with chapter timestamps
"""

from __future__ import annotations

from .captions import make_srt
from .notes_agent import ChapterMark, extract_chapters, generate_show_notes
from .packager import PackageResult, package_episode

__all__ = [
    "make_srt",
    "ChapterMark",
    "extract_chapters",
    "generate_show_notes",
    "PackageResult",
    "package_episode",
]
