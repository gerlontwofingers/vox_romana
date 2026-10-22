"""Tests for vox_romana.core.crew — crew identity, episodic memory, and consolidation."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.core.crew import (
    CONSOLIDATION_THRESHOLD,
    KEEP_HIGH_SALIENCE,
    KEEP_RECENT,
    CrewMember,
    CrewProfile,
    EpisodicEntry,
    _prune_short_term,
    _strip_json_fences,
    compute_salience,
    consolidate_if_needed,
    load_crew_member,
    write_episodic_entry,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_entry(
    crew_id: str = "demo",
    show_type: str = "debate",
    title: str = "Test Show",
    salience: float = 0.0,
) -> EpisodicEntry:
    return EpisodicEntry(
        run_id=f"{crew_id}_run",
        show_type=show_type,
        title=title,
        date="2026-05-10T00:00:00Z",
        salience_score=salience,
        summary="A test run.",
    )


def _write_entries(path: Path, entries: list[EpisodicEntry]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e.to_dict()) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CrewProfile
# ─────────────────────────────────────────────────────────────────────────────


class TestCrewProfile:
    def test_format_for_prompt_includes_name(self):
        profile = CrewProfile(name="Alice", role="director")
        text = profile.format_for_prompt()
        assert "CREW IDENTITY: Alice" in text

    def test_format_for_prompt_omits_empty_fields(self):
        profile = CrewProfile(name="Bob", role="director")
        text = profile.format_for_prompt()
        assert "Aesthetic philosophy" not in text
        assert "Threshold calibration" not in text

    def test_format_for_prompt_includes_nonempty_fields(self):
        profile = CrewProfile(
            name="Carol",
            role="director",
            aesthetic_philosophy="Minimalist",
            pacing_approach="Slow burn",
        )
        text = profile.format_for_prompt()
        assert "Aesthetic philosophy: Minimalist" in text
        assert "Pacing approach: Slow burn" in text

    def test_from_yaml_loads_all_fields(self, tmp_path):
        yaml_content = (
            "name: Marcus Webb\n"
            "role: director\n"
            "aesthetic_philosophy: Stark clarity\n"
            "stylistic_tendencies: Short punchy sentences\n"
            "pacing_approach: Builds tension\n"
            "conflict_stance: Welcomes confrontation\n"
            "threshold_calibration: Conservative\n"
        )
        p = tmp_path / "profile.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        profile = CrewProfile.from_yaml(p)
        assert profile.name == "Marcus Webb"
        assert profile.role == "director"
        assert profile.aesthetic_philosophy == "Stark clarity"
        assert profile.stylistic_tendencies == "Short punchy sentences"
        assert profile.pacing_approach == "Builds tension"
        assert profile.conflict_stance == "Welcomes confrontation"
        assert profile.threshold_calibration == "Conservative"

    def test_from_yaml_defaults_name_to_parent_dir(self, tmp_path):
        crew_dir = tmp_path / "my_director"
        crew_dir.mkdir()
        p = crew_dir / "profile.yaml"
        p.write_text("role: director\n", encoding="utf-8")
        profile = CrewProfile.from_yaml(p)
        assert profile.name == "my_director"

    def test_from_yaml_empty_file_produces_defaults(self, tmp_path):
        crew_dir = tmp_path / "anon"
        crew_dir.mkdir()
        p = crew_dir / "profile.yaml"
        p.write_text("", encoding="utf-8")
        profile = CrewProfile.from_yaml(p)
        assert profile.role == "director"
        assert profile.aesthetic_philosophy == ""


# ─────────────────────────────────────────────────────────────────────────────
# CrewMember
# ─────────────────────────────────────────────────────────────────────────────


class TestCrewMember:
    def test_profile_text_returns_empty_when_no_profile(self):
        member = CrewMember(crew_id="x", role="director", profile=None)
        assert member.profile_text() == ""

    def test_profile_text_returns_formatted_profile(self):
        profile = CrewProfile(
            name="Dana", role="director", aesthetic_philosophy="Raw"
        )
        member = CrewMember(crew_id="dana", role="director", profile=profile)
        text = member.profile_text()
        assert "CREW IDENTITY: Dana" in text

    def test_load_crew_member_with_missing_yaml(self, tmp_path):
        member = load_crew_member("ghost", "director", crew_dir=tmp_path)
        assert member.crew_id == "ghost"
        assert member.role == "director"
        assert member.profile is None

    def test_load_crew_member_with_valid_yaml(self, tmp_path):
        crew_dir = tmp_path / "anna"
        crew_dir.mkdir()
        (crew_dir / "profile.yaml").write_text(
            "name: Anna\nrole: factchecker\n", encoding="utf-8"
        )
        member = load_crew_member("anna", "factchecker", crew_dir=tmp_path)
        assert member.profile is not None
        assert member.profile.name == "Anna"

    def test_load_crew_member_with_malformed_yaml(self, tmp_path):
        crew_dir = tmp_path / "broken"
        crew_dir.mkdir()
        (crew_dir / "profile.yaml").write_text(
            "name: [not\nclosed\n", encoding="utf-8"
        )
        # Must not raise; returns member with profile=None
        member = load_crew_member("broken", "director", crew_dir=tmp_path)
        assert member.profile is None


# ─────────────────────────────────────────────────────────────────────────────
# compute_salience
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeSalience:
    def test_empty_log_gives_zero(self):
        assert compute_salience([]) == 0.0

    def test_issue_report_adds_one(self):
        log = [{"event_type": "issue_report"}]
        assert compute_salience(log) == 1.0

    def test_retries_exhausted_directive_adds_two(self):
        log = [
            {
                "event_type": "agent_message",
                "channel": "directive",
                "payload": {"reason": "retries_exhausted"},
            }
        ]
        assert compute_salience(log) == 2.0

    def test_director_note_broadcast_adds_one(self):
        log = [
            {
                "event_type": "agent_message",
                "channel": "broadcast",
                "payload": {"phase": "director_note"},
            }
        ]
        assert compute_salience(log) == 1.0

    def test_fabricated_challenge_adds_one(self):
        log = [
            {
                "event_type": "agent_message",
                "channel": "challenge",
                "result": {"verdict": "fabricated"},
            }
        ]
        assert compute_salience(log) == 1.0

    def test_verified_challenge_adds_nothing(self):
        log = [
            {
                "event_type": "agent_message",
                "channel": "challenge",
                "result": {"verdict": "verified"},
            }
        ]
        assert compute_salience(log) == 0.0

    def test_salience_caps_at_ten(self):
        log = [{"event_type": "issue_report"}] * 20
        assert compute_salience(log) == 10.0

    def test_additive_scoring(self):
        log = [
            {"event_type": "issue_report"},
            {
                "event_type": "agent_message",
                "channel": "directive",
                "payload": {"reason": "retries_exhausted"},
            },
            {
                "event_type": "agent_message",
                "channel": "challenge",
                "result": {"verdict": "fabricated"},
            },
        ]
        assert compute_salience(log) == 4.0  # 1+2+1


# ─────────────────────────────────────────────────────────────────────────────
# write_episodic_entry
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteEpisodicEntry:
    def test_creates_jsonl_file(self, tmp_path):
        entry = _make_entry()
        write_episodic_entry("demo", entry, crew_dir=tmp_path)
        path = tmp_path / "demo" / "short_term.jsonl"
        assert path.exists()

    def test_written_entry_is_valid_json(self, tmp_path):
        entry = _make_entry(salience=3.5)
        write_episodic_entry("demo", entry, crew_dir=tmp_path)
        path = tmp_path / "demo" / "short_term.jsonl"
        data = json.loads(path.read_text().strip())
        assert data["salience_score"] == 3.5

    def test_appends_multiple_entries(self, tmp_path):
        for i in range(3):
            entry = _make_entry(title=f"Show {i}", salience=float(i))
            write_episodic_entry("demo", entry, crew_dir=tmp_path)
        path = tmp_path / "demo" / "short_term.jsonl"
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_creates_crew_directory(self, tmp_path):
        entry = _make_entry()
        write_episodic_entry("new_crew_member", entry, crew_dir=tmp_path)
        assert (tmp_path / "new_crew_member").is_dir()


# ─────────────────────────────────────────────────────────────────────────────
# _prune_short_term
# ─────────────────────────────────────────────────────────────────────────────


class TestPruneShortTerm:
    def _entries(self, n: int, *, saliences: list[float] | None = None) -> list[dict]:
        saliences = saliences or [0.0] * n
        return [
            {
                "run_id": f"run_{i}",
                "salience_score": saliences[i] if i < len(saliences) else 0.0,
            }
            for i in range(n)
        ]

    def test_empty_list_returns_empty(self):
        assert _prune_short_term([]) == []

    def test_small_list_keeps_all(self):
        entries = self._entries(3)
        assert _prune_short_term(entries) == entries

    def test_keeps_high_salience(self):
        entries = self._entries(
            10, saliences=[0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 9.0, 10.0, 7.0, 0.0]
        )
        kept = _prune_short_term(
            entries, keep_high_salience=3, keep_recent=0
        )
        run_ids = {e["run_id"] for e in kept}
        # Entries with salience 8, 9, 10 → run_5, run_6, run_7
        assert "run_6" in run_ids  # salience 9
        assert "run_7" in run_ids  # salience 10

    def test_keeps_most_recent(self):
        entries = self._entries(10)
        kept = _prune_short_term(entries, keep_high_salience=0, keep_recent=3)
        run_ids = [e["run_id"] for e in kept]
        # Most recent are run_7, run_8, run_9
        assert "run_9" in run_ids
        assert "run_8" in run_ids
        assert "run_7" in run_ids

    def test_no_duplicates_when_high_salience_and_recent_overlap(self):
        entries = self._entries(10, saliences=[0.0] * 7 + [10.0, 10.0, 10.0])
        kept = _prune_short_term(entries, keep_high_salience=3, keep_recent=3)
        run_ids = [e["run_id"] for e in kept]
        assert len(run_ids) == len(set(run_ids))


# ─────────────────────────────────────────────────────────────────────────────
# _strip_json_fences
# ─────────────────────────────────────────────────────────────────────────────


class TestStripJsonFences:
    def test_plain_json_unchanged(self):
        text = '{"key": "value"}'
        assert _strip_json_fences(text) == text

    def test_strips_backtick_fence(self):
        text = "```json\n{\"key\": 1}\n```"
        result = _strip_json_fences(text)
        assert result == '{"key": 1}'

    def test_strips_plain_fence(self):
        text = "```\n{\"a\": 2}\n```"
        result = _strip_json_fences(text)
        assert result == '{"a": 2}'


# ─────────────────────────────────────────────────────────────────────────────
# consolidate_if_needed
# ─────────────────────────────────────────────────────────────────────────────


class TestConsolidateIfNeeded:
    def _fill_short_term(
        self, crew_dir: Path, crew_id: str, count: int
    ) -> None:
        for i in range(count):
            entry = _make_entry(crew_id=crew_id, salience=float(i % 5))
            write_episodic_entry(crew_id, entry, crew_dir=crew_dir)

    def test_below_threshold_returns_false(self, tmp_path):
        self._fill_short_term(tmp_path, "demo", CONSOLIDATION_THRESHOLD - 1)
        client = MagicMock()
        result = consolidate_if_needed(
            "demo", client=client, crew_dir=tmp_path, threshold=CONSOLIDATION_THRESHOLD
        )
        assert result is False
        client.chat.assert_not_called()

    def test_at_threshold_calls_llm_and_returns_true(self, tmp_path):
        self._fill_short_term(tmp_path, "demo", CONSOLIDATION_THRESHOLD)
        mock_response = json.dumps(
            {
                "generalizations": [
                    {
                        "text": "I have found that debates go long when topics are abstract.",
                        "show_tags": ["debate"],
                        "derived_from_runs": CONSOLIDATION_THRESHOLD,
                    }
                ]
            }
        )
        client = MagicMock()
        client.chat.return_value = mock_response
        result = consolidate_if_needed(
            "demo", client=client, crew_dir=tmp_path, threshold=CONSOLIDATION_THRESHOLD
        )
        assert result is True
        client.chat.assert_called_once()

    def test_consolidation_writes_long_term_jsonl(self, tmp_path):
        self._fill_short_term(tmp_path, "demo", CONSOLIDATION_THRESHOLD)
        mock_response = json.dumps(
            {
                "generalizations": [
                    {
                        "text": "Keep debates short.",
                        "show_tags": ["debate"],
                        "derived_from_runs": 5,
                    }
                ]
            }
        )
        client = MagicMock()
        client.chat.return_value = mock_response
        consolidate_if_needed(
            "demo", client=client, crew_dir=tmp_path, threshold=CONSOLIDATION_THRESHOLD
        )
        long_path = tmp_path / "demo" / "long_term.jsonl"
        assert long_path.exists()
        data = json.loads(long_path.read_text().strip())
        assert "Keep debates short." in data["text"]

    def test_consolidation_prunes_short_term(self, tmp_path):
        self._fill_short_term(tmp_path, "demo", CONSOLIDATION_THRESHOLD)
        client = MagicMock()
        client.chat.return_value = json.dumps(
            {"generalizations": [{"text": "Lesson.", "show_tags": [], "derived_from_runs": 1}]}
        )
        consolidate_if_needed(
            "demo", client=client, crew_dir=tmp_path, threshold=CONSOLIDATION_THRESHOLD
        )
        short_path = tmp_path / "demo" / "short_term.jsonl"
        lines = [l for l in short_path.read_text().splitlines() if l.strip()]
        assert len(lines) < CONSOLIDATION_THRESHOLD
        assert len(lines) <= KEEP_HIGH_SALIENCE + KEEP_RECENT

    def test_llm_failure_returns_false_and_preserves_short_term(self, tmp_path):
        self._fill_short_term(tmp_path, "demo", CONSOLIDATION_THRESHOLD)
        client = MagicMock()
        client.chat.side_effect = RuntimeError("LLM unavailable")
        result = consolidate_if_needed(
            "demo", client=client, crew_dir=tmp_path, threshold=CONSOLIDATION_THRESHOLD
        )
        assert result is False
        # Short_term should be untouched
        short_path = tmp_path / "demo" / "short_term.jsonl"
        lines = [l for l in short_path.read_text().splitlines() if l.strip()]
        assert len(lines) == CONSOLIDATION_THRESHOLD

    def test_empty_generalizations_returns_false(self, tmp_path):
        self._fill_short_term(tmp_path, "demo", CONSOLIDATION_THRESHOLD)
        client = MagicMock()
        client.chat.return_value = json.dumps({"generalizations": []})
        result = consolidate_if_needed(
            "demo", client=client, crew_dir=tmp_path, threshold=CONSOLIDATION_THRESHOLD
        )
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# BaseConductor — _augmented_director_system_prompt
# ─────────────────────────────────────────────────────────────────────────────


class TestAugmentedDirectorSystemPrompt:
    """Integration-level tests via BaseConductor (using a concrete subclass stub)."""

    def _make_conductor(self, crew_members=None):
        """Build a minimal conductor with mocked infrastructure."""
        from unittest.mock import MagicMock

        from vox_romana.core.conductor import BaseConductor
        from vox_romana.core.config import ContentConfig

        # Build a concrete subclass that fills in the abstract methods
        class _StubConductor(BaseConductor):
            def _build_agents(self):
                pass
            def _run_turn_loop(self):
                return []
            def _package_result(self, raw):
                from vox_romana.core.conductor import ConductorResult
                return ConductorResult(config=self.config)

        config = MagicMock(spec=ContentConfig)
        config.content_type = "podcast"
        config.target_words = None
        config.target_duration_minutes = None
        config.estimated_duration_minutes = None
        config.tolerance_percent = 10
        client = MagicMock()
        return _StubConductor(config, client=client, crew_members=crew_members)

    def test_no_crew_returns_base_prompt_unchanged(self):
        conductor = self._make_conductor()
        base = "You are a podcast Director. Respond with JSON only."
        result = conductor._augmented_director_system_prompt(base)
        assert result == base

    def test_director_profile_prepended_to_base(self):
        profile = CrewProfile(
            name="Marcus Webb",
            role="director",
            aesthetic_philosophy="Minimalist and sharp",
        )
        member = CrewMember(crew_id="marcus_webb", role="director", profile=profile)
        conductor = self._make_conductor(crew_members=[member])
        base = "You are a podcast Director. Respond with JSON only."
        result = conductor._augmented_director_system_prompt(base)
        assert result.startswith("CREW IDENTITY: Marcus Webb")
        assert base in result

    def test_factchecker_profile_does_not_appear_in_director_prompt(self):
        profile = CrewProfile(name="FC Agent", role="factchecker")
        member = CrewMember(crew_id="fc", role="factchecker", profile=profile)
        conductor = self._make_conductor(crew_members=[member])
        base = "You are a podcast Director. Respond with JSON only."
        result = conductor._augmented_director_system_prompt(base)
        # factchecker profile should NOT appear in director prompt
        assert "CREW IDENTITY" not in result
        assert result == base

    def test_empty_profile_fields_do_not_add_content(self):
        profile = CrewProfile(name="Minimal", role="director")
        member = CrewMember(crew_id="min", role="director", profile=profile)
        conductor = self._make_conductor(crew_members=[member])
        base = "You are a debate director. Respond with JSON only."
        result = conductor._augmented_director_system_prompt(base)
        # Profile with only a name should still get prepended
        assert "CREW IDENTITY: Minimal" in result

    def test_factchecker_profile_is_captured_for_show_conductors_to_forward(self):
        """C-3: show conductors read self._crew_profiles.get('factchecker', '')
        when constructing FactChecker.build_default() — confirm the data is
        actually there to forward, with the threshold_calibration field that
        matters for fact-check behaviour."""
        profile = CrewProfile(
            name="FC Agent", role="factchecker", threshold_calibration="lenient",
        )
        member = CrewMember(crew_id="fc", role="factchecker", profile=profile)
        conductor = self._make_conductor(crew_members=[member])
        text = conductor._crew_profiles.get("factchecker", "")
        assert text.startswith("CREW IDENTITY: FC Agent")
        assert "Threshold calibration: lenient" in text

    def test_no_factchecker_crew_member_returns_empty_string(self):
        conductor = self._make_conductor()
        assert conductor._crew_profiles.get("factchecker", "") == ""


# ─────────────────────────────────────────────────────────────────────────────
# C-4: runners._write_crew_episodic — end-of-run episodic write wiring
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteCrewEpisodic:
    """vox_romana.cli.runners._write_crew_episodic — the glue between a
    finished run's production_log and crew.write_episodic_entry()."""

    def _make_result(self, *, production_log=None, events=None, llm_calls=3):
        return SimpleNamespace(
            production_log=production_log or [],
            debate_events=events if events is not None else [{"text": "a"}, {"text": "b"}],
            total_llm_calls=llm_calls,
            production_log_path="",
        )

    def _make_config(self, *, content_type="debate", title="Test Debate"):
        return SimpleNamespace(content_type=content_type, title=title)

    def test_no_crew_members_is_a_noop(self, tmp_path):
        from vox_romana.cli.runners import _write_crew_episodic

        args = SimpleNamespace(crew_dir=str(tmp_path))
        _write_crew_episodic(
            self._make_result(), self._make_config(), [], MagicMock(), args,
        )
        assert not list(tmp_path.glob("**/*.jsonl"))

    def test_writes_entry_with_computed_salience(self, tmp_path):
        from vox_romana.cli.runners import _write_crew_episodic

        production_log = [{"event_type": "issue_report"}]
        result = self._make_result(production_log=production_log)
        config = self._make_config()
        member = CrewMember(crew_id="marcus_webb", role="director", profile=None)
        args = SimpleNamespace(crew_dir=str(tmp_path))

        _write_crew_episodic(result, config, [member], MagicMock(), args)

        short_term = tmp_path / "marcus_webb" / "short_term.jsonl"
        assert short_term.exists()
        entries = [json.loads(l) for l in short_term.read_text().splitlines() if l.strip()]
        assert len(entries) == 1
        assert entries[0]["salience_score"] == compute_salience(production_log)
        assert entries[0]["show_type"] == "debate"
        assert "2 events" in entries[0]["summary"]
        assert "1 issue report" in entries[0]["summary"]

    def test_writes_one_entry_per_crew_member(self, tmp_path):
        from vox_romana.cli.runners import _write_crew_episodic

        result = self._make_result()
        config = self._make_config()
        members = [
            CrewMember(crew_id="director_a", role="director", profile=None),
            CrewMember(crew_id="fc_b", role="factchecker", profile=None),
        ]
        args = SimpleNamespace(crew_dir=str(tmp_path))

        _write_crew_episodic(result, config, members, MagicMock(), args)

        assert (tmp_path / "director_a" / "short_term.jsonl").exists()
        assert (tmp_path / "fc_b" / "short_term.jsonl").exists()

    def test_write_failure_is_logged_not_raised(self, tmp_path):
        """A crew write error must never abort an otherwise-successful run."""
        from vox_romana.cli.runners import _write_crew_episodic

        result = self._make_result()
        config = self._make_config()
        member = CrewMember(crew_id="broken", role="director", profile=None)
        args = SimpleNamespace(crew_dir=str(tmp_path))

        with patch(
            "vox_romana.cli.runners.write_episodic_entry",
            side_effect=OSError("disk full"),
        ):
            # Must not raise.
            _write_crew_episodic(result, config, [member], MagicMock(), args)

    def test_consolidation_triggered_when_called(self, tmp_path):
        """consolidate_if_needed() must actually be invoked (below-threshold
        is fine — the point is it's wired, not skipped)."""
        from vox_romana.cli.runners import _write_crew_episodic

        result = self._make_result()
        config = self._make_config()
        member = CrewMember(crew_id="demo", role="director", profile=None)
        args = SimpleNamespace(crew_dir=str(tmp_path))

        with patch("vox_romana.cli.runners.consolidate_if_needed") as mock_consolidate:
            _write_crew_episodic(result, config, [member], MagicMock(), args)
            mock_consolidate.assert_called_once()
            assert mock_consolidate.call_args.args[0] == "demo"
            assert mock_consolidate.call_args.kwargs["crew_dir"] == tmp_path
