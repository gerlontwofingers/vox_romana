"""Tests for vox_romana.studio.instantiator (shot 5.2-b).

Covers the staged orchestration: 3 LLM calls → catalog create_series →
optional episode-1 draft, with stubbed client + catalog.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

from vox_romana.core.audience import AudienceProfile
from vox_romana.studio import StudioInstantiator, StudioInstantiation


def _meta_json():
    return json.dumps({
        "title": "Signal & Static",
        "show_type": "podcast",
        "tone": "warm and skeptical",
        "series_id": "signal-and-static",
        "premise": "Tech culture for millennials.",
    })


def _bible_json():
    return json.dumps({"bible": "Tone: curious. Motifs: systems, trade-offs."})


def _audience_json():
    return json.dumps({
        "age_range": "25-40",
        "education_level": "undergraduate",
        "vocabulary_level": "educated",
        "preferred_tone": "conversational",
        "prior_knowledge": "Consumer tech familiar.",
    })


def _episode_json():
    return json.dumps({
        "title": "Episode 1",
        "angle": "Why we trust machines.",
        "hooks": ["A cold open about a wrong GPS."],
        "guests": [],
        "estimated_duration": 30,
        "brief": "A podcast episode about trust in technology.",
    })


class _FakeCatalog:
    """Stub catalog with the 3 methods the instantiator touches."""

    def __init__(self):
        self.created = []
        self._episode_number = 1

    def create_series(self, **kwargs):
        self.created.append(kwargs)
        return "uuid1234567890"

    def next_episode_number(self, series_id):
        return self._episode_number

    def get_recent_episode_summaries(self, series_id, n=5):
        return []


class TestStudioInstantiator:
    def _make(self, *, draft_episode=True):
        client = Mock()
        client.chat.side_effect = [_meta_json(), _bible_json(), _audience_json()]
        catalog = _FakeCatalog()
        instantiator = StudioInstantiator(client, catalog=catalog)
        return instantiator, client, catalog

    def test_instantiate_runs_three_stages(self):
        instantiator, client, catalog = self._make(draft_episode=False)
        result = instantiator.instantiate(
            "A tech culture podcast for millennials.",
            draft_episode=False,
        )

        assert isinstance(result, StudioInstantiation)
        assert client.chat.call_count == 3
        # Meta
        assert result.meta.title == "Signal & Static"
        assert result.meta.show_type == "podcast"
        # Bible
        assert result.bible.startswith("Tone: curious")
        # Audience
        assert isinstance(result.audience, AudienceProfile)
        assert result.audience.vocabulary_level == "educated"
        # Catalog
        assert result.series_id == "uuid1234567890"
        assert len(catalog.created) == 1
        created = catalog.created[0]
        assert created["title"] == "Signal & Static"
        assert created["show_type"] == "podcast"
        assert created["bible"].startswith("Tone: curious")
        assert created["audience_profile"].vocabulary_level == "educated"
        # No episode draft requested.
        assert result.episode_brief is None

    def test_instantiate_drafts_episode(self):
        instantiator, client, catalog = self._make(draft_episode=True)
        # 3 staged calls + 1 episode-planner call.
        client.chat.side_effect = [
            _meta_json(), _bible_json(), _audience_json(), _episode_json(),
        ]
        result = instantiator.instantiate("A tech culture podcast.")

        assert client.chat.call_count == 4
        assert result.episode_brief is not None
        assert result.episode_brief.title == "Episode 1"
        assert "trust in technology" in result.episode_brief.prompt_text

    def test_draft_episode_failure_is_nonfatal(self):
        instantiator, client, catalog = self._make(draft_episode=True)
        client.chat.side_effect = [
            _meta_json(), _bible_json(), _audience_json(), "not json",
        ]
        result = instantiator.instantiate("A tech culture podcast.")
        assert result.episode_brief is None  # graceful degradation
        assert result.series_id == "uuid1234567890"

    def test_meta_failure_raises(self):
        instantiator, client, catalog = self._make(draft_episode=False)
        client.chat.side_effect = ["not json"]
        import pytest
        with pytest.raises(ValueError):
            instantiator.instantiate("A tech culture podcast.")
