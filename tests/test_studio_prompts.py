"""Tests for vox_romana.studio.prompts (shot 5.2-a).

Covers the three staged prompts (meta / bible / audience) and their
parsers, including validation/fallback behavior.
"""

from __future__ import annotations

import json

from vox_romana.core.audience import AudienceProfile
from vox_romana.studio import (
    StudioMeta,
    build_audience_user_prompt,
    build_bible_user_prompt,
    build_meta_user_prompt,
    parse_audience,
    parse_bible,
    parse_meta,
)


def _meta_json(**overrides):
    data = {
        "title": "Signal & Static",
        "show_type": "podcast",
        "tone": "warm and skeptical",
        "series_id": "Signal-and-Static!",  # deliberate: needs slugify
        "premise": "A weekly podcast unpacking tech culture for millennials.",
    }
    data.update(overrides)
    return json.dumps(data)


def _bible_json():
    return json.dumps({"bible": "Tone: curious, never preachy.\nMotifs: systems, trade-offs."})


def _audience_json(**overrides):
    data = {
        "age_range": "25-40",
        "education_level": "undergraduate",
        "vocabulary_level": "educated",
        "preferred_tone": "conversational, data-light",
        "prior_knowledge": "Familiar with consumer tech, not deep on policy.",
    }
    data.update(overrides)
    return json.dumps(data)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — meta
# ═══════════════════════════════════════════════════════════════════════════

class TestParseMeta:
    def test_parses_clean_json(self):
        meta = parse_meta(_meta_json())
        assert isinstance(meta, StudioMeta)
        assert meta.title == "Signal & Static"
        assert meta.show_type == "podcast"
        assert meta.tone == "warm and skeptical"
        assert meta.premise.startswith("A weekly podcast")

    def test_slugifies_series_id(self):
        meta = parse_meta(_meta_json())
        assert meta.series_id == "signal-and-static"

    def test_unknown_show_type_falls_back_to_podcast(self):
        meta = parse_meta(_meta_json(show_type="variety_show"))
        assert meta.show_type == "podcast"

    def test_missing_series_id_derives_from_title(self):
        data = json.loads(_meta_json())
        data.pop("series_id")
        meta = parse_meta(json.dumps(data))
        assert meta.series_id == "signal-static"

    def test_missing_title_falls_back(self):
        meta = parse_meta(json.dumps({"show_type": "news"}))
        assert meta.title == "Untitled Studio"

    def test_parses_markdown_fenced(self):
        meta = parse_meta("```json\n" + _meta_json() + "\n```")
        assert meta.title == "Signal & Static"

    def test_invalid_json_raises(self):
        import pytest
        with pytest.raises(ValueError):
            parse_meta("no json here")

    def test_build_user_prompt_contains_brief(self):
        prompt = build_meta_user_prompt("a true crime podcast for Brazilian millennials")
        assert "a true crime podcast for Brazilian millennials" in prompt


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — bible
# ═══════════════════════════════════════════════════════════════════════════

class TestParseBible:
    def test_parses_bible(self):
        bible = parse_bible(_bible_json())
        assert bible.startswith("Tone: curious")

    def test_parses_markdown_fenced(self):
        bible = parse_bible("```json\n" + _bible_json() + "\n```")
        assert "Tone: curious" in bible

    def test_missing_bible_returns_empty(self):
        assert parse_bible(json.dumps({})) == ""

    def test_build_user_prompt_contains_meta(self):
        meta = parse_meta(_meta_json())
        prompt = build_bible_user_prompt(meta)
        assert "Signal & Static" in prompt
        assert "podcast" in prompt


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 — audience
# ═══════════════════════════════════════════════════════════════════════════

class TestParseAudience:
    def test_parses_full_profile(self):
        ap = parse_audience(_audience_json())
        assert isinstance(ap, AudienceProfile)
        assert ap.age_range == "25-40"
        assert ap.education_level == "undergraduate"
        assert ap.vocabulary_level == "educated"
        assert ap.preferred_tone == "conversational, data-light"

    def test_invalid_vocab_falls_back_to_conversational(self):
        ap = parse_audience(_audience_json(vocabulary_level="gobbledygook"))
        assert ap.vocabulary_level == "conversational"

    def test_invalid_education_falls_back_to_none(self):
        ap = parse_audience(_audience_json(education_level="space-cadet"))
        assert ap.education_level is None

    def test_minimal_profile_defaults(self):
        ap = parse_audience(json.dumps({}))
        assert ap.vocabulary_level == "conversational"
        assert ap.age_range is None
        assert ap.prior_knowledge is None

    def test_build_user_prompt_contains_meta(self):
        meta = parse_meta(_meta_json())
        prompt = build_audience_user_prompt(meta)
        assert "Signal & Static" in prompt
        assert "podcast" in prompt
