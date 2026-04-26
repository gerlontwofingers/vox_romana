"""Flexible loader — converts text files, directories, and vox_romana JSON into
SpeechSegment lists suitable for the emotion_flow pipeline.

Supported inputs:
    .txt file          — split by double-newline (paragraph = segment)
    directory          — loads all .txt files, one speech_id per file
    vox_romana JSON    — parses events[], segments[].events[], or chunks[].events[]
                         for the specified speaker/leader name
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import SpeechSegment

logger = logging.getLogger(__name__)

_MIN_CHARS: int = 20

_SPEECH_KINDS: frozenset[str] = frozenset({
    "utterance", "narration", "speech",
})


class FlexibleLoader:
    """Factory for creating SpeechSegment lists from various input formats."""

    @staticmethod
    def load(
        source: str | Path,
        leader: str | None = None,
        *,
        min_chars: int = _MIN_CHARS,
    ) -> list[SpeechSegment]:
        """Load speech segments from a file or directory.

        Args:
            source:    Path to a .txt file, .json file, or directory of .txt files.
            leader:    Speaker name filter (case-insensitive substring match).
                       Required when loading a vox_romana JSON with multiple speakers.
            min_chars: Minimum segment character length; shorter segments are skipped.

        Returns:
            List of SpeechSegment objects. source_lang and translation_confidence
            are not yet populated (pending TranslatorAgent); defaults to "" and 0.0.

        Raises:
            FileNotFoundError: If source does not exist.
            ValueError:        If a vox_romana JSON is given without a leader name
                               and multiple speakers are present.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")

        if path.is_dir():
            return FlexibleLoader._load_directory(path, min_chars=min_chars)
        elif path.suffix.lower() == ".json":
            return FlexibleLoader._load_vox_romana_json(
                path, leader=leader, min_chars=min_chars
            )
        else:
            return FlexibleLoader._load_text_file(
                path, speech_id=path.stem, min_chars=min_chars
            )

    @staticmethod
    def _load_text_file(
        path: Path,
        speech_id: str,
        min_chars: int,
    ) -> list[SpeechSegment]:
        text = path.read_text(encoding="utf-8", errors="replace")
        raw_paragraphs = text.split("\n\n")
        segments: list[SpeechSegment] = []
        idx = 0
        for para in raw_paragraphs:
            cleaned = para.strip()
            if len(cleaned) < min_chars:
                continue
            segments.append(SpeechSegment(
                text=cleaned,
                original_text=cleaned,
                source_lang="",
                translation_confidence=0.0,
                translation_flagged=False,
                segment_index=idx,
                speech_id=speech_id,
            ))
            idx += 1
        logger.debug("Loaded %d segments from %s", len(segments), path)
        return segments

    @staticmethod
    def _load_directory(path: Path, min_chars: int) -> list[SpeechSegment]:
        txt_files = sorted(path.glob("*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt files found in directory: {path}")
        all_segments: list[SpeechSegment] = []
        global_idx = 0
        for txt in txt_files:
            file_segments = FlexibleLoader._load_text_file(
                txt, speech_id=txt.stem, min_chars=min_chars
            )
            for seg in file_segments:
                all_segments.append(SpeechSegment(
                    text=seg.text,
                    original_text=seg.original_text,
                    source_lang=seg.source_lang,
                    translation_confidence=seg.translation_confidence,
                    translation_flagged=seg.translation_flagged,
                    segment_index=global_idx,
                    speech_id=seg.speech_id,
                ))
                global_idx += 1
        logger.info(
            "Loaded %d segments from %d files in %s",
            len(all_segments), len(txt_files), path,
        )
        return all_segments

    @staticmethod
    def _load_vox_romana_json(
        path: Path,
        leader: str | None,
        min_chars: int,
    ) -> list[SpeechSegment]:
        data = json.loads(path.read_text(encoding="utf-8"))
        events = _extract_events(data)

        if leader is None:
            speakers = {
                e.get("speaker_name", "")
                for e in events
                if e.get("event_kind") in _SPEECH_KINDS
                and e.get("speaker_name")
            }
            if len(speakers) == 1:
                leader = next(iter(speakers))
            else:
                raise ValueError(
                    f"vox_romana JSON has multiple speakers "
                    f"({', '.join(sorted(speakers))}). "
                    "Pass --leader to filter."
                )

        leader_lower = leader.lower()
        segments: list[SpeechSegment] = []
        speech_id = path.stem
        idx = 0
        for event in events:
            if event.get("event_kind") not in _SPEECH_KINDS:
                continue
            speaker = event.get("speaker_name", "")
            if leader_lower not in speaker.lower():
                continue
            text = (event.get("text") or "").strip()
            if len(text) < min_chars:
                continue
            segments.append(SpeechSegment(
                text=text,
                original_text=text,
                source_lang="",
                translation_confidence=0.0,
                translation_flagged=False,
                segment_index=idx,
                speech_id=speech_id,
            ))
            idx += 1
        logger.info(
            "Loaded %d segments for leader '%s' from %s",
            len(segments), leader, path,
        )
        return segments


def _extract_events(data: dict) -> list[dict]:
    """Unify vox_romana's three event layouts into a flat list."""
    if "events" in data:
        return data["events"]
    collected: list[dict] = []
    for seg in data.get("segments", []):
        collected.extend(seg.get("events", []))
    for chunk in data.get("chunks", []):
        collected.extend(chunk.get("events", []))
    return collected
