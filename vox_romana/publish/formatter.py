"""Formatters — convert raw show JSON into structured ``FormattedDocument``s.

One formatter per show type.  Each follows a real industry convention so
that a human picking up the printed output recognises the form:

* **DebateFormatter**   → stage-play / screenplay layout.
                         Speaker labels in ALL CAPS followed by a colon,
                         dialogue indented underneath, stage directions
                         centred in (parentheses).  Multi-chunk debates
                         become numbered acts.

* **NewsFormatter**     → broadcast-news rundown layout.
                         Each segment gets a SLUG header, then a labelled
                         block: ``ANCHOR ON CAM:`` for anchor reads,
                         ``VO:`` for voiceover narration, ``SOT:`` for
                         sound-on-tape (interviews / packages),
                         ``REPORTER:`` for field reports, ``TAG:`` for
                         the wrap-up line.

* **PodcastFormatter**  → Rev / Otter style transcript.
                         ``Speaker Name:`` on its own line, dialogue
                         paragraph beneath.  Non-verbal cues become
                         bracketed inline tags ([LAUGHTER], [CROSSTALK]).

* **AudiobookFormatter**→ book chapter layout.
                         Chapter headings, prose paragraphs, no speaker
                         labels for narration.  Dialogue events get
                         minimal ``"Quoted text," said Name.`` framing.
                         Scene breaks rendered as ``* * *``.

Use :func:`format_document` to auto-pick a formatter from the JSON shape.
"""

from __future__ import annotations

from typing import Any

from .models import FormattedDocument, Line, Section


# ─────────────────────────────────────────────────────────────────────────────
# Section-label i18n
# ─────────────────────────────────────────────────────────────────────────────

# Maps a BCP-47 language code to localized labels for the structural
# section headings that appear in formatted output.  Only the "act" and
# "chapter" labels have meaningful translations; broadcast labels like
# "SLUG" and "PKG" are industry-standard across languages.
_SECTION_LABELS: dict[str, dict[str, str]] = {
    "en": {"act": "ACT", "chapter": "Chapter", "segment": "Segment"},
    "es": {"act": "ACTO", "chapter": "Capítulo", "segment": "Segmento"},
    "fr": {"act": "ACTE", "chapter": "Chapitre", "segment": "Segment"},
    "de": {"act": "AKT",  "chapter": "Kapitel",  "segment": "Abschnitt"},
    "pt": {"act": "ATO",  "chapter": "Capítulo", "segment": "Segmento"},
    "it": {"act": "ATTO", "chapter": "Capitolo", "segment": "Segmento"},
    "nl": {"act": "BEDRIJF", "chapter": "Hoofdstuk", "segment": "Segment"},
    "pl": {"act": "AKT",  "chapter": "Rozdział",  "segment": "Segment"},
    "ru": {"act": "АКТ",  "chapter": "Глава",     "segment": "Сегмент"},
    "ja": {"act": "幕",    "chapter": "章",         "segment": "セグメント"},
    "zh": {"act": "幕",    "chapter": "章",         "segment": "部分"},
    "ko": {"act": "막",    "chapter": "장",         "segment": "세그먼트"},
    "ar": {"act": "فصل",  "chapter": "فصل",        "segment": "مقطع"},
}

_DEFAULT_LABELS: dict[str, str] = _SECTION_LABELS["en"]


def _get_labels(language: str) -> dict[str, str]:
    """Return section labels for *language*, falling back to English."""
    return _SECTION_LABELS.get(language, _DEFAULT_LABELS)


# ─────────────────────────────────────────────────────────────────────────────
# Show-type detection
# ─────────────────────────────────────────────────────────────────────────────

# Identifier keys on the top-level JSON object, in priority order.
# We check the most specific marker first so a debate that happens to
# carry a ``segments`` field (it doesn't, but defensively) still routes
# to the debate formatter.
_TYPE_MARKERS = (
    ("debate_id", "debate"),
    ("audiobook_id", "audiobook"),
    ("podcast_id", "podcast"),
    ("show_id", "news"),
)


def detect_show_type(data: dict[str, Any]) -> str:
    """Infer the show type from a raw show-JSON dict.

    Args:
        data: Parsed JSON object as written by the show serializers.

    Returns:
        One of ``"debate"``, ``"news"``, ``"podcast"``, ``"audiobook"``.

    Raises:
        ValueError: If no recognised identifier key is present.
    """
    for key, kind in _TYPE_MARKERS:
        if key in data:
            return kind
    raise ValueError(
        "Cannot detect show type: JSON has none of "
        f"{[k for k, _ in _TYPE_MARKERS]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_list(value: Any) -> list[str]:
    """Coerce ``value`` to a list of stripped non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def _join_oneline(value: Any, sep: str = ", ") -> str:
    """Join a list-like value into a single comma-separated line."""
    items = _norm_list(value)
    return sep.join(items)


def _nonverbal_to_directions(event: dict[str, Any]) -> list[str]:
    """Pull non-verbal cues + tone/pace/volume into stage-direction strings.

    Returns a list of human-readable cues suitable for parenthetical
    stage directions.  Empty list if there's nothing notable.
    """
    cues: list[str] = []
    cues.extend(_norm_list(event.get("nonverbal")))
    return cues


def _delivery_note(event: dict[str, Any]) -> str:
    """Build a one-line delivery note from tone/pace/volume.

    Returns an empty string if no delivery data is present.
    """
    bits: list[str] = []
    tones = _norm_list(event.get("tone"))
    if tones:
        bits.append(", ".join(tones))
    pace = (event.get("pace") or "").strip()
    if pace and pace.lower() not in ("medium", "moderate"):
        bits.append(f"{pace} pace")
    volume = (event.get("volume") or "").strip()
    if volume and volume.lower() not in ("moderate",):
        bits.append(volume)
    return "; ".join(bits)


def _is_spoken(event: dict[str, Any]) -> bool:
    """True if this event has spoken text we should render."""
    text = (event.get("text") or "").strip()
    return bool(text)


# ─────────────────────────────────────────────────────────────────────────────
# Base formatter
# ─────────────────────────────────────────────────────────────────────────────

class DocumentFormatter:
    """Base class.  Subclasses implement :meth:`format`."""

    show_type: str = ""

    def format(self, data: dict[str, Any]) -> FormattedDocument:
        """Convert a parsed show-JSON dict into a ``FormattedDocument``."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Debate — stage play / screenplay layout
# ─────────────────────────────────────────────────────────────────────────────

class DebateFormatter(DocumentFormatter):
    """Renders a debate as a stage-play / screenplay script.

    Conventions used:

    * Title page with cast list and setting.
    * Speaker label in ALL CAPS, on its own line, followed by a colon.
    * Dialogue paragraph below the label.
    * Non-verbal cues as parenthetical stage directions on their own
      line, indented and italicised by HTML renderers.
    * Each chunk in a multi-chunk debate becomes ``ACT N``.  A single
      chunk uses the single-act convention with no act header.
    """

    show_type = "debate"

    def format(self, data: dict[str, Any]) -> FormattedDocument:
        title = (data.get("title") or "Untitled Debate").strip()
        topic = (data.get("topic") or "").strip()
        debate_format = (data.get("debate_format") or "").strip()
        setting = (data.get("setting") or "").strip()
        participants = _norm_list(data.get("participants"))
        moderators = _norm_list(data.get("moderators"))
        hosts = _norm_list(data.get("hosts"))
        language = (data.get("language") or "en").strip()
        labels = _get_labels(language)

        # ── Front matter ────────────────────────────────────────────
        front: list[Line] = []
        if topic:
            front.append(Line("metadata", f"Topic:  {topic}"))
        if debate_format:
            front.append(Line("metadata", f"Format: {debate_format}"))
        if setting:
            front.append(Line("metadata", f"Setting: {setting}"))
        cast_bits: list[str] = []
        if participants:
            cast_bits.append("Participants: " + ", ".join(participants))
        if moderators:
            cast_bits.append("Moderator"
                             + ("s" if len(moderators) > 1 else "")
                             + ": " + ", ".join(moderators))
        if hosts:
            cast_bits.append("Host"
                             + ("s" if len(hosts) > 1 else "")
                             + ": " + ", ".join(hosts))
        for bit in cast_bits:
            front.append(Line("metadata", bit))

        # ── Body sections (one per chunk, or a single section) ──────
        chunks = data.get("chunks") or []
        sections: list[Section] = []

        if len(chunks) > 1:
            # Multi-act layout.
            for chunk in chunks:
                act_n = chunk.get("chunk_order") or (len(sections) + 1)
                sec = Section(title=f"{labels['act']} {act_n}", kind="chunk")
                self._render_events(chunk.get("events", []), sec)
                sections.append(sec)
        else:
            # Single section using the flat top-level events list, falling
            # back to the single chunk's events if present.
            events = data.get("events")
            if not events and chunks:
                events = chunks[0].get("events", [])
            sec = Section(title="", kind="chunk")
            self._render_events(events or [], sec)
            sections.append(sec)

        return FormattedDocument(
            show_type=self.show_type,
            title=title,
            language=language,
            front_matter=front,
            sections=sections,
            meta={
                "topic": topic,
                "participants": participants,
                "moderators": moderators,
                "hosts": hosts,
                "format": debate_format,
                "setting": setting,
            },
        )

    @staticmethod
    def _render_events(events: list[dict[str, Any]], sec: Section) -> None:
        """Render each event as speaker / direction / dialogue lines."""
        last_speaker: str | None = None
        for event in events:
            kind = (event.get("event_kind") or "utterance").lower()
            text = (event.get("text") or "").strip()

            # Pause-only / break events become a horizontal rule when long.
            if kind == "break":
                sec.lines.append(Line("rule"))
                last_speaker = None
                continue

            if not _is_spoken(event):
                continue

            speaker = (event.get("speaker_name") or "Unknown").strip()
            role = (event.get("speaker_role") or "").lower()

            # Leading non-verbal cues become a stage direction line above
            # the speaker label, *only* if the speaker is changing or
            # there's no prior speaker (avoids visual clutter).
            cues = _nonverbal_to_directions(event)
            if cues and speaker != last_speaker:
                sec.lines.append(Line("direction", f"({'; '.join(cues)})"))
                cues = []  # consumed

            # Narration and host events render as italic prose without a
            # speaker label, matching screenplay convention for narrator
            # voice-over.
            if kind == "narration" or role == "host":
                if last_speaker is not None:
                    sec.lines.append(Line("blank"))
                sec.lines.append(Line("narration", text))
                sec.lines.append(Line("blank"))
                last_speaker = None
                continue

            # Standard dialogue: speaker label + dialogue.
            if speaker != last_speaker:
                if last_speaker is not None:
                    sec.lines.append(Line("blank"))
                sec.lines.append(Line("speaker", speaker.upper() + ":"))

            # Inline (mid-utterance) cues that survived above.
            for cue in cues:
                sec.lines.append(Line("direction", f"({cue})"))

            sec.lines.append(Line("dialogue", text))

            # Trailing non-verbal cues we couldn't attach (e.g. short
            # gestures after the line) are dropped — too noisy for a
            # readable script and already implied by the event flow.

            last_speaker = speaker


# ─────────────────────────────────────────────────────────────────────────────
# News — broadcast rundown layout
# ─────────────────────────────────────────────────────────────────────────────

# Mapping from segment_type → labelled cue used to introduce the segment's
# body block.  These are real broadcast newsroom labels.
_SEG_TYPE_LABEL = {
    "anchor_read": "ANCHOR ON CAM",
    "package":     "PKG",
    "panel":       "PANEL",
    "breaking":    "BREAKING",
    "interview":   "SOT",
    "kicker":      "KICKER",
}

# Mapping from speaker_role → spoken-line label inside a segment body.
_ROLE_LABEL = {
    "anchor":   "ANCHOR",
    "reporter": "REPORTER",
    "host":     "HOST",
    "guest":    "GUEST",
    "panelist": "PANEL",
    "narrator": "VO",
}


class NewsFormatter(DocumentFormatter):
    """Renders a news show as a broadcast rundown / script.

    Each segment becomes a slugged block with a labelled body.  The
    layout mirrors what a TV-news producer would print for the on-air
    talent: a SLUG line, source/anchor cue, then the spoken copy.
    """

    show_type = "news"

    def format(self, data: dict[str, Any]) -> FormattedDocument:
        title = (data.get("title") or "Untitled News Show").strip()
        topic = (data.get("topic") or "").strip()
        tone = (data.get("tone") or "").strip()
        anchors = _norm_list(data.get("anchors"))
        hosts = _norm_list(data.get("hosts"))
        guests = _norm_list(data.get("guests"))
        language = (data.get("language") or "en").strip()

        front: list[Line] = []
        if topic:
            front.append(Line("metadata", f"Topic: {topic}"))
        if tone:
            front.append(Line("metadata", f"Tone:  {tone}"))
        if anchors:
            front.append(Line("metadata", "Anchor"
                              + ("s" if len(anchors) > 1 else "")
                              + ": " + ", ".join(anchors)))
        if hosts:
            front.append(Line("metadata", "Host"
                              + ("s" if len(hosts) > 1 else "")
                              + ": " + ", ".join(hosts)))
        if guests:
            front.append(Line("metadata", "Guest"
                              + ("s" if len(guests) > 1 else "")
                              + ": " + ", ".join(guests)))

        sections: list[Section] = []
        for seg in data.get("segments") or []:
            order = seg.get("segment_order")
            order_str = f"{int(order) + 1:02d}" if order is not None else "??"
            headline = (seg.get("headline") or "(no headline)").strip()
            seg_type = (seg.get("segment_type") or "").lower()
            seg_label = _SEG_TYPE_LABEL.get(seg_type, seg_type.upper() or "SEGMENT")

            sec = Section(
                title=f"SLUG {order_str} — {headline.upper()}",
                kind="segment",
            )

            # Cue line (segment type + anchor name if known).
            cue_bits = [seg_label]
            anchor_name = (seg.get("anchor_name") or "").strip()
            if anchor_name:
                cue_bits.append(f"Anchor: {anchor_name}")
            seg_tone = (seg.get("tone") or "").strip()
            if seg_tone:
                cue_bits.append(f"Tone: {seg_tone}")
            sec.lines.append(Line("metadata", "  •  ".join(cue_bits)))
            sec.lines.append(Line("blank"))

            self._render_events(seg.get("events") or [], sec)
            sections.append(sec)

        return FormattedDocument(
            show_type=self.show_type,
            title=title,
            language=language,
            front_matter=front,
            sections=sections,
            meta={
                "topic": topic,
                "tone": tone,
                "anchors": anchors,
                "hosts": hosts,
                "guests": guests,
            },
        )

    @staticmethod
    def _render_events(events: list[dict[str, Any]], sec: Section) -> None:
        last_speaker: str | None = None
        for event in events:
            if not _is_spoken(event):
                continue
            text = event["text"].strip()
            speaker = (event.get("speaker_name") or "").strip()
            role = (event.get("speaker_role") or "").lower()
            label = _ROLE_LABEL.get(role, "")
            label_str = label or (speaker.upper() if speaker else "VOICE")

            # News uses ROLE-LED labels (ANCHOR ON CAM, REPORTER, etc.)
            # rather than per-speaker labels, so consecutive lines from
            # the same anchor merge into one paragraph block.  Display
            # convention: NAME (ROLE) — as on a TelePrompTer cue card.
            if speaker and label:
                full_label = f"{speaker.upper()} ({label})"
            elif speaker:
                full_label = speaker.upper()
            else:
                full_label = label_str

            if speaker != last_speaker:
                if last_speaker is not None:
                    sec.lines.append(Line("blank"))
                sec.lines.append(Line("speaker", full_label + ":"))

            sec.lines.append(Line("dialogue", text))
            last_speaker = speaker


# ─────────────────────────────────────────────────────────────────────────────
# Podcast — Rev/Otter transcript layout
# ─────────────────────────────────────────────────────────────────────────────

# Map of common nonverbal phrases → bracketed cue tokens used inline in
# transcripts.  Anything not in this map falls back to the raw cue text
# wrapped in [SQUARE BRACKETS].
_NONVERBAL_INLINE = {
    "laughs":     "[LAUGHTER]",
    "laughing":   "[LAUGHTER]",
    "chuckles":   "[CHUCKLES]",
    "sighs":      "[SIGHS]",
    "pause":      "[PAUSE]",
    "crosstalk":  "[CROSSTALK]",
}


class PodcastFormatter(DocumentFormatter):
    """Renders a podcast as a Rev.com / Otter.ai-style transcript."""

    show_type = "podcast"

    def format(self, data: dict[str, Any]) -> FormattedDocument:
        title = (data.get("title") or "Untitled Podcast").strip()
        show_name = (data.get("show_name") or "").strip()
        topic = (data.get("topic") or "").strip()
        tone = (data.get("tone") or "").strip()
        hosts = _norm_list(data.get("hosts"))
        language = (data.get("language") or "en").strip()
        labels = _get_labels(language)

        front: list[Line] = []
        if show_name:
            front.append(Line("metadata", f"Show:  {show_name}"))
        if topic:
            front.append(Line("metadata", f"Topic: {topic}"))
        if tone:
            front.append(Line("metadata", f"Tone:  {tone}"))
        if hosts:
            front.append(Line("metadata", "Host"
                              + ("s" if len(hosts) > 1 else "")
                              + ": " + ", ".join(hosts)))

        sections: list[Section] = []
        segments = data.get("segments") or []
        if segments:
            for seg in segments:
                order = seg.get("segment_order")
                order_str = (f"{labels['segment']} {int(order):02d}"
                             if order is not None else labels["segment"])
                seg_type = (seg.get("segment_type") or "").strip()
                seg_topic = (seg.get("topic") or "").strip()
                heading_bits = [order_str]
                if seg_type:
                    heading_bits.append(f"({seg_type})")
                if seg_topic:
                    heading_bits.append("— " + seg_topic)
                sec = Section(title=" ".join(heading_bits), kind="segment")
                self._render_events(seg.get("events") or [], sec)
                sections.append(sec)
        else:
            # Flat-events podcast (no segment grouping).
            sec = Section(title="", kind="segment")
            self._render_events(data.get("events") or [], sec)
            sections.append(sec)

        return FormattedDocument(
            show_type=self.show_type,
            title=title,
            language=language,
            front_matter=front,
            sections=sections,
            meta={
                "show_name": show_name,
                "topic": topic,
                "hosts": hosts,
            },
        )

    @staticmethod
    def _inline_cues(event: dict[str, Any]) -> str:
        """Build a leading bracketed-cue string for inline transcript cues."""
        cues = _nonverbal_to_directions(event)
        if not cues:
            return ""
        tokens: list[str] = []
        for cue in cues:
            key = cue.lower().strip().rstrip(".")
            if key in _NONVERBAL_INLINE:
                tokens.append(_NONVERBAL_INLINE[key])
            else:
                tokens.append("[" + cue.upper() + "]")
        return " ".join(tokens)

    def _render_events(
        self, events: list[dict[str, Any]], sec: Section,
    ) -> None:
        last_speaker: str | None = None
        for event in events:
            if not _is_spoken(event):
                continue
            text = event["text"].strip()
            speaker = (event.get("speaker_name") or "Unknown").strip()
            cues = self._inline_cues(event)

            if speaker != last_speaker:
                if last_speaker is not None:
                    sec.lines.append(Line("blank"))
                sec.lines.append(Line("speaker", f"{speaker}:"))

            line_text = (cues + " " + text).strip() if cues else text
            sec.lines.append(Line("dialogue", line_text))
            last_speaker = speaker


# ─────────────────────────────────────────────────────────────────────────────
# Audiobook — book chapter layout
# ─────────────────────────────────────────────────────────────────────────────

# Speaker roles whose events should render as plain prose without any
# attribution or quotation marks.
_NARRATIVE_ROLES = {"narrator", "host"}


class AudiobookFormatter(DocumentFormatter):
    """Renders an audiobook as book chapters.

    Narrator events become plain prose paragraphs.  Character dialogue
    becomes minimally-attributed quoted speech::

        "Get out," said Frank.

    Where ``Frank`` is the speaker_name.  Polishing of attribution
    variety (replied / muttered / whispered) is the LLM polisher's job;
    this stage uses a single neutral verb so the deterministic output
    is stable.
    """

    show_type = "audiobook"

    def format(self, data: dict[str, Any]) -> FormattedDocument:
        title = (data.get("title") or "Untitled Audiobook").strip()
        narrator = (data.get("narrator") or "").strip()
        genre = (data.get("genre") or "").strip()
        tone = (data.get("tone") or "").strip()
        characters = _norm_list(data.get("characters"))
        language = (data.get("language") or "en").strip()
        labels = _get_labels(language)

        front: list[Line] = []
        if genre:
            front.append(Line("metadata", f"Genre:    {genre}"))
        if tone:
            front.append(Line("metadata", f"Tone:     {tone}"))
        if narrator:
            front.append(Line("metadata", f"Narrator: {narrator}"))
        if characters:
            front.append(Line("metadata",
                              "Characters: " + ", ".join(characters)))

        sections: list[Section] = []
        chapters = data.get("chapters") or []
        if chapters:
            for ch in chapters:
                order = ch.get("chapter_order") or (len(sections) + 1)
                ch_title = (ch.get("title") or "").strip()
                heading = f"{labels['chapter']} {order}"
                if ch_title:
                    heading = f"{heading}: {ch_title}"
                sec = Section(title=heading, kind="chapter")
                self._render_events(ch.get("events") or [], sec, narrator)
                sections.append(sec)
        else:
            # Flat-events audiobook (no chapter grouping).
            sec = Section(title="", kind="chapter")
            self._render_events(
                data.get("events") or [], sec, narrator,
            )
            sections.append(sec)

        return FormattedDocument(
            show_type=self.show_type,
            title=title,
            language=language,
            front_matter=front,
            sections=sections,
            meta={
                "narrator": narrator,
                "characters": characters,
                "genre": genre,
            },
        )

    @staticmethod
    def _render_events(
        events: list[dict[str, Any]],
        sec: Section,
        narrator: str,
    ) -> None:
        for event in events:
            if not _is_spoken(event):
                # break events become scene breaks.
                if (event.get("event_kind") or "").lower() == "break":
                    sec.lines.append(Line("rule"))
                continue

            text = event["text"].strip()
            speaker = (event.get("speaker_name") or "").strip()
            role = (event.get("speaker_role") or "").lower()

            is_narration = (
                role in _NARRATIVE_ROLES
                or (narrator and speaker == narrator)
                or not speaker
            )

            if is_narration:
                sec.lines.append(Line("narration", text))
            else:
                # Minimal attribution.  Strip outer quotes if the LLM
                # already provided them so we don't double-quote, then
                # apply standard fiction punctuation:
                #   "Get out!" said Frank.        (terminal !/?/…)
                #   "He left," said Frank.        (otherwise)
                stripped = text.strip().strip('"').strip("“”").rstrip()
                if stripped.endswith(("!", "?", "…")):
                    quoted = f'"{stripped}"'
                else:
                    # Drop a trailing period so we can use the comma form.
                    if stripped.endswith("."):
                        stripped = stripped[:-1].rstrip()
                    quoted = f'"{stripped},"'
                sec.lines.append(
                    Line("dialogue", f"{quoted} said {speaker}.")
                )

            sec.lines.append(Line("blank"))


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatch
# ─────────────────────────────────────────────────────────────────────────────

_FORMATTERS: dict[str, type[DocumentFormatter]] = {
    "debate":    DebateFormatter,
    "news":      NewsFormatter,
    "podcast":   PodcastFormatter,
    "audiobook": AudiobookFormatter,
}


def format_document(data: dict[str, Any]) -> FormattedDocument:
    """Auto-detect show type and run the matching formatter.

    Args:
        data: Parsed show JSON.

    Returns:
        A ``FormattedDocument`` ready for rendering to TXT or PDF.

    Raises:
        ValueError: If the show type cannot be detected.
    """
    show_type = detect_show_type(data)
    formatter_cls = _FORMATTERS[show_type]
    return formatter_cls().format(data)
