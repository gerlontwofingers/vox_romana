"""Tests for Chunk 5: per-character lens routing, position-aware context,
warm-open first-appearance nudge, and speaker attribution in _build_turn_context.

All tests use lightweight fakes — no real LLM calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from vox_romana.core.lens import LLMLensRack, build_default_rack
from vox_romana.shows.debates.conductor import DebateConductor
from vox_romana.shows.debates.config import DebateConfig, ParticipantProfile


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(n_participants: int = 2, turns: int = 10) -> DebateConfig:
    """Build a minimal DebateConfig with n_participants and no moderator/host."""
    names = ["Alice", "Bob", "Carol", "Dave"][:n_participants]
    participants = []
    for i, name in enumerate(names):
        participants.append(
            ParticipantProfile(
                name=name,
                role="panelist",
                side="pro" if i % 2 == 0 else "con",
                voice_gender="female" if i % 2 == 0 else "male",
            )
        )
    return DebateConfig(
        topic="Is the sky blue?",
        brief="A debate on sky colour",
        participants=participants,
        target_turns=turns,
        fact_check="off",
    )


def _make_mock_client() -> MagicMock:
    """A fake LLMClient whose .chat() always returns valid JSON."""
    client = MagicMock()
    client.chat.return_value = (
        '{"text": "That is a fair point.", "tone": "calm", "volume": "moderate",'
        ' "pace": "medium", "target": "all", "pause_before": "none",'
        ' "pause_after": "none", "interruption": false, "nonverbal": []}'
    )
    return client


def _make_conductor(
    config: DebateConfig | None = None,
    lens_rack: LLMLensRack | None = None,
) -> DebateConductor:
    config = config or _make_config()
    conductor = DebateConductor(config, client=_make_mock_client())
    conductor._lens_rack = lens_rack
    return conductor


# ─────────────────────────────────────────────────────────────────────────────
# 1. lens_rack wiring — per-character client selection
# ─────────────────────────────────────────────────────────────────────────────


def test_no_lens_rack_uses_shared_client() -> None:
    """When lens_rack is None, all sessions use the shared default client."""
    config = _make_config()
    shared_client = _make_mock_client()
    conductor = DebateConductor(config, client=shared_client)
    conductor._build_agents()

    for name in ["Alice", "Bob"]:
        sess = conductor._agent_sessions[name]
        assert sess._client is shared_client


def test_lens_rack_without_character_route_uses_shared_client() -> None:
    """Rack with no character routes → all sessions still use shared client."""
    config = _make_config()
    shared_client = _make_mock_client()
    rack = build_default_rack()  # no character-specific routes

    conductor = DebateConductor(config, client=shared_client, lens_rack=rack)
    conductor._build_agents()

    for name in ["Alice", "Bob"]:
        sess = conductor._agent_sessions[name]
        assert sess._client is shared_client


def test_lens_rack_character_specific_route_overrides_client() -> None:
    """A character:<Name> route gives that character their own lens client."""
    config = _make_config()
    shared_client = _make_mock_client()

    rack = build_default_rack()
    alice_client = _make_mock_client()
    # Inject a fake lens for Alice
    from vox_romana.core.lens import LLMLens
    alice_lens = LLMLens(
        name="alice-lens",
        backend="local",
        model="test-model",
        factory=lambda: alice_client,
    )
    rack._lenses["alice-lens"] = alice_lens
    rack._routes["character:alice"] = "alice-lens"

    conductor = DebateConductor(config, client=shared_client, lens_rack=rack)
    conductor._build_agents()

    assert conductor._agent_sessions["Alice"]._client is alice_client
    # Bob is unaffected
    assert conductor._agent_sessions["Bob"]._client is shared_client


def test_lens_rack_catch_all_character_route() -> None:
    """A bare 'character' catch-all route applies to all participants."""
    config = _make_config()
    shared_client = _make_mock_client()
    char_client = _make_mock_client()

    rack = build_default_rack()
    from vox_romana.core.lens import LLMLens
    char_lens = LLMLens(
        name="char-catch-all",
        backend="local",
        model="tiny-model",
        factory=lambda: char_client,
    )
    rack._lenses["char-catch-all"] = char_lens
    rack._routes["character"] = "char-catch-all"

    conductor = DebateConductor(config, client=shared_client, lens_rack=rack)
    conductor._build_agents()

    for name in ["Alice", "Bob"]:
        assert conductor._agent_sessions[name]._client is char_client


# ─────────────────────────────────────────────────────────────────────────────
# 2. Position-aware debate phase in _build_turn_context
# ─────────────────────────────────────────────────────────────────────────────


def _ctx(conductor, *, turn_number, total_turns, speaker_name="Alice",
         speaker_role="participant", wrapping_up=False):
    remaining = max(0, total_turns - turn_number + 1)
    return conductor._build_turn_context(
        speaker_name=speaker_name,
        speaker_role=speaker_role,
        turn_number=turn_number,
        wrapping_up=wrapping_up,
        remaining_turns=remaining,
        target_words=0,
    )


def test_opening_phase_injected_for_early_turns() -> None:
    conductor = _make_conductor(_make_config(turns=20))
    conductor._build_agents()
    context = _ctx(conductor, turn_number=2, total_turns=20)
    assert "OPENING" in context


def test_development_phase_injected_for_mid_turns() -> None:
    conductor = _make_conductor(_make_config(turns=20))
    conductor._build_agents()
    context = _ctx(conductor, turn_number=10, total_turns=20)
    assert "DEVELOPMENT" in context


def test_late_phase_injected_for_late_turns() -> None:
    conductor = _make_conductor(_make_config(turns=20))
    conductor._build_agents()
    context = _ctx(conductor, turn_number=16, total_turns=20)
    assert "LATE" in context


def test_no_phase_hint_for_moderator() -> None:
    conductor = _make_conductor(_make_config(turns=20))
    conductor._build_agents()
    context = _ctx(conductor, turn_number=10, total_turns=20,
                   speaker_name="Bob", speaker_role="moderator")
    assert "DEBATE PHASE" not in context


def test_no_phase_hint_when_wrapping_up() -> None:
    conductor = _make_conductor(_make_config(turns=10))
    conductor._build_agents()
    context = _ctx(conductor, turn_number=10, total_turns=10,
                   wrapping_up=True)
    assert "DEBATE PHASE" not in context


# ─────────────────────────────────────────────────────────────────────────────
# 3. Warm-open first-appearance nudge
# ─────────────────────────────────────────────────────────────────────────────


def test_warm_open_nudge_on_first_turn() -> None:
    conductor = _make_conductor(_make_config(turns=10))
    conductor._build_agents()
    # Alice has not spoken yet → should see FIRST APPEARANCE nudge
    context = _ctx(conductor, turn_number=2, total_turns=10,
                   speaker_name="Alice", speaker_role="participant")
    assert "FIRST APPEARANCE" in context


def test_warm_open_nudge_absent_on_second_turn() -> None:
    conductor = _make_conductor(_make_config(turns=10))
    conductor._build_agents()
    conductor._has_spoken.add("Alice")  # simulate Alice already spoke
    context = _ctx(conductor, turn_number=4, total_turns=10,
                   speaker_name="Alice", speaker_role="participant")
    assert "FIRST APPEARANCE" not in context


def test_warm_open_nudge_absent_for_moderator() -> None:
    conductor = _make_conductor(_make_config(turns=10))
    conductor._build_agents()
    context = _ctx(conductor, turn_number=2, total_turns=10,
                   speaker_name="Mod", speaker_role="moderator")
    assert "FIRST APPEARANCE" not in context


# ─────────────────────────────────────────────────────────────────────────────
# 4. Speaker attribution nudge in turn instruction
# ─────────────────────────────────────────────────────────────────────────────


def test_attribution_nudge_includes_last_speaker_name() -> None:
    conductor = _make_conductor(_make_config(turns=10))
    conductor._build_agents()
    # Seed transcript so Bob just spoke
    conductor._transcript = [
        {"speaker": "Bob", "role": "participant", "text": "I think not.", "turn": 1}
    ]
    conductor._has_spoken.update(["Alice", "Bob"])
    context = _ctx(conductor, turn_number=2, total_turns=10,
                   speaker_name="Alice", speaker_role="participant")
    assert "Address Bob by name" in context


def test_no_attribution_nudge_on_first_turn() -> None:
    """When transcript is empty there's no one to attribute yet."""
    conductor = _make_conductor(_make_config(turns=10))
    conductor._build_agents()
    # transcript is empty
    context = _ctx(conductor, turn_number=1, total_turns=10,
                   speaker_name="Alice", speaker_role="participant")
    assert "Address" not in context or "by name" not in context


def test_no_attribution_nudge_when_responding_to_self() -> None:
    """No 'address X by name' when the last speaker is the same speaker."""
    conductor = _make_conductor(_make_config(turns=10))
    conductor._build_agents()
    conductor._transcript = [
        {"speaker": "Alice", "role": "participant", "text": "I stand by it.", "turn": 3}
    ]
    conductor._has_spoken.add("Alice")
    context = _ctx(conductor, turn_number=4, total_turns=10,
                   speaker_name="Alice", speaker_role="participant")
    assert "Address Alice by name" not in context


# ─────────────────────────────────────────────────────────────────────────────
# 5. _has_spoken reset on each run()
# ─────────────────────────────────────────────────────────────────────────────


def test_has_spoken_reset_in_build_agents() -> None:
    conductor = _make_conductor(_make_config(turns=4))
    conductor._has_spoken = {"Alice", "Bob"}  # dirty state from a prior run
    conductor._build_agents()
    assert conductor._has_spoken == set()


# ─────────────────────────────────────────────────────────────────────────────
# 6. runners.py passes lens_rack to DebateConductor
# ─────────────────────────────────────────────────────────────────────────────


def test_run_debate_passes_lens_rack_to_conductor() -> None:
    """_run_debate should forward args.lens_rack to DebateConductor constructor.

    DebateBootstrap and DebateConductor are lazy-imported inside _run_debate,
    so we patch them at their source modules rather than through runners.
    """
    import argparse
    from unittest.mock import patch, MagicMock
    from vox_romana.cli import runners

    fake_rack = MagicMock()
    args = argparse.Namespace(
        turns=None, tone=None, debate_format=None, max_retries=None,
        allow_bootstrap_fabrications=False, duration=None, tolerance=None,
        checkpoint=None, content_mode="auto", reference_material=None,
        fact_check="off", series_id=None, tag=None, name=None,
        output=None, script_output=None, quiet=True, engine="kokoro",
        tts_workers=1, llm_lens=None, llm_backend="auto",
        root_lang="en", out_lang=None, keep_translated=False,
        translation_model=None, university_session=None,
        publish=False, lens_rack=fake_rack,
    )

    fake_config = _make_config(turns=2)
    fake_result = MagicMock()
    fake_result.debate_events = []
    fake_result.total_llm_calls = 0

    captured_kwargs: dict = {}

    class CapturingConductor:
        def __init__(self, config, *, client=None, lens_rack=None, **kw):
            captured_kwargs["lens_rack"] = lens_rack
            self._length_warnings = []

        def run(self):
            return fake_result

    class FakeBootstrap:
        def __init__(self, **kw):
            pass

        def from_brief(self, brief, **kw):
            return fake_config

    # Both DebateBootstrap and DebateConductor are imported inside _run_debate
    # using ``from ... import``, so we patch at source module level.
    with patch("vox_romana.shows.debates.bootstrap.DebateBootstrap", FakeBootstrap), \
         patch("vox_romana.shows.debates.conductor.DebateConductor", CapturingConductor), \
         patch.object(runners, "_build_output_path", return_value="/tmp/x.json"), \
         patch.object(runners, "_build_bootstrap_kwargs", return_value={}), \
         patch.object(runners, "_build_checkpoint_kwargs", return_value={}), \
         patch.object(runners, "_read_reference_material", return_value=None), \
         patch.object(runners, "_resolve_duration", return_value={}), \
         patch.object(runners, "_count_words_in_events", return_value=0), \
         patch.object(runners, "_log_length_summary"), \
         patch.object(runners, "estimate_total_words", return_value=1000), \
         patch("vox_romana.shows.debates.serializer.DebateSerializer"):
        runners._run_debate("Is the sky blue?", _make_mock_client(), args)

    assert captured_kwargs.get("lens_rack") is fake_rack


def test_write_production_log_persists_jsonl(tmp_path) -> None:
    from vox_romana.cli import runners

    output_path = tmp_path / "episode.json"
    result = MagicMock()
    result.production_log = [
        {"event_type": "agent_message", "channel": "broadcast"},
        {"event_type": "issue_report", "status": "queued"},
    ]
    result.production_log_path = None

    runners._write_production_log(output_path, result)

    log_path = tmp_path / "episode_production_log.jsonl"
    assert log_path.exists()
    assert result.production_log_path == str(log_path)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == result.production_log


def test_run_debate_writes_production_log() -> None:
    """_run_debate should persist the result production log next to the JSON."""
    import argparse
    from unittest.mock import patch, MagicMock
    from vox_romana.cli import runners

    args = argparse.Namespace(
        turns=None, tone=None, debate_format=None, max_retries=None,
        allow_bootstrap_fabrications=False, duration=None, tolerance=None,
        checkpoint=None, content_mode="auto", reference_material=None,
        fact_check="off", series_id=None, tag=None, name=None,
        output=None, script_output=None, quiet=True, engine="kokoro",
        tts_workers=1, llm_lens=None, llm_backend="auto",
        root_lang="en", out_lang=None, keep_translated=False,
        translation_model=None, university_session=None,
        publish=False, lens_rack=None,
    )

    fake_config = _make_config(turns=2)
    fake_result = MagicMock()
    fake_result.debate_events = []
    fake_result.total_llm_calls = 0
    fake_result.production_log = [{"event_type": "agent_message"}]

    class CapturingConductor:
        def __init__(self, config, *, client=None, lens_rack=None, **kw):
            self._length_warnings = []

        def run(self):
            return fake_result

    class FakeBootstrap:
        def __init__(self, **kw):
            pass

        def from_brief(self, brief, **kw):
            return fake_config

    with patch("vox_romana.shows.debates.bootstrap.DebateBootstrap", FakeBootstrap), \
         patch("vox_romana.shows.debates.conductor.DebateConductor", CapturingConductor), \
         patch.object(runners, "_build_output_path", return_value="/tmp/x.json"), \
         patch.object(runners, "_build_bootstrap_kwargs", return_value={}), \
         patch.object(runners, "_build_checkpoint_kwargs", return_value={}), \
         patch.object(runners, "_read_reference_material", return_value=None), \
         patch.object(runners, "_resolve_duration", return_value={}), \
         patch.object(runners, "_count_words_in_events", return_value=0), \
         patch.object(runners, "_log_length_summary"), \
         patch.object(runners, "_write_production_log") as mock_write_log, \
         patch.object(runners, "estimate_total_words", return_value=1000), \
         patch("vox_romana.shows.debates.serializer.DebateSerializer"):
        runners._run_debate("Is the sky blue?", _make_mock_client(), args)

    mock_write_log.assert_called_once_with("/tmp/x.json", fake_result)
