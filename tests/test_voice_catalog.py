"""Tests for voice_catalog.py — usage-weighted fallback and variety.

Covers the new behaviour introduced by the am_adam variety fix:

1. match_voice() zero-score branch — prefers least-used voices
2. match_voice() position tiebreaker — same counts, different positions differ
3. match_voice() with partial usage — heavily-used voice skipped in favour of fresh one
4. match_voice() with empty usage_counts — still returns a result (no crash)
5. match_voice() tag branch unchanged — winning tag match beats usage count
6. match_voices_for_cast() threads usage_counts through to each match_voice call
7. 5-male cast variety — am_adam appears exactly once, not for all 5 speakers
8. Usage-weighted reorder — most-used voice sorted last in the candidate pool

These tests are isolated from the renderer — they test voice_catalog.py directly.
"""

import pytest

from vox_romana.tts.voice_catalog import (
    KOKORO_VOICE_CATALOG,
    match_voice,
    match_voices_for_cast,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

ALL_MALE_KOKORO = [
    "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]

ALL_FEMALE_KOKORO = [
    "af_heart", "af_alloy", "af_aoede", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river",
    "af_sarah", "af_sky",
    "bf_emma", "bf_isabella", "bf_lily",
]


# ═══════════════════════════════════════════════════════════════════════════
# match_voice() — zero-score fallback with usage_counts
# ═══════════════════════════════════════════════════════════════════════════

class TestMatchVoiceZeroScoreFallback:
    def test_empty_hint_returns_a_voice(self):
        """An empty hint should still return a voice (no crash)."""
        result = match_voice("", gender="male", usage_counts={})
        assert result is not None

    def test_empty_hint_empty_usage_returns_am_adam_at_position_0(self):
        """With zero usage and position 0, am_adam is the catalog-order first male."""
        result = match_voice("", gender="male", position=0, usage_counts={})
        assert result == "am_adam"

    def test_position_tiebreaker_differs_across_positions(self):
        """With equal usage counts, different positions should get different voices."""
        v0 = match_voice("", gender="male", position=0, usage_counts={})
        v1 = match_voice("", gender="male", position=1, usage_counts={})
        v2 = match_voice("", gender="male", position=2, usage_counts={})
        assert v0 != v1, f"position 0 and 1 both got {v0}"
        assert v1 != v2, f"position 1 and 2 both got {v1}"

    def test_heavily_used_voice_not_chosen_first(self):
        """A voice with a high usage count should not be chosen first."""
        # Make am_adam heavily used.
        usage = {"am_adam": 100}
        result = match_voice("", gender="male", position=0, usage_counts=usage)
        assert result != "am_adam", f"Expected a different voice, got am_adam"

    def test_least_used_voice_chosen_first(self):
        """The voice with the lowest count should win the zero-score contest."""
        # Make everything used except am_liam.
        usage = {v: 5 for v in ALL_MALE_KOKORO if v != "am_liam"}
        result = match_voice("", gender="male", position=0, usage_counts=usage)
        assert result == "am_liam"

    def test_usage_counts_none_falls_back_to_catalog_order(self):
        """When usage_counts is None, behaviour is legacy (catalog insertion order)."""
        # No usage_counts argument — am_adam is first in the catalog.
        result = match_voice("", gender="male", usage_counts=None)
        assert result == "am_adam"

    def test_zero_score_female_prefers_least_used(self):
        """Same least-used logic applies to female voices."""
        usage = {v: 3 for v in ALL_FEMALE_KOKORO if v != "af_jessica"}
        result = match_voice("", gender="female", position=0, usage_counts=usage)
        assert result == "af_jessica"

    def test_exclude_set_respected_in_zero_score_branch(self):
        """Excluded voices must not be returned even in the zero-score branch."""
        result = match_voice(
            "",
            gender="male",
            exclude={"am_adam"},
            usage_counts={},
        )
        assert result is not None
        assert result != "am_adam"


# ═══════════════════════════════════════════════════════════════════════════
# match_voice() — tag-match branch should not be affected by usage_counts
# ═══════════════════════════════════════════════════════════════════════════

class TestMatchVoiceTagBranchUnchanged:
    def test_tag_match_wins_over_usage_count(self):
        """A good tag match should always win regardless of usage count."""
        # Use a hint that combines several tags unique to bf_emma so its
        # score is unambiguously higher than any other female voice.
        # ("composed british polished moderator" scores 4 for bf_emma,
        # ≤1 for every other female voice in the catalog.)
        matching_hint = "composed british polished moderator"

        high_usage = {"bf_emma": 999}
        result = match_voice(
            matching_hint,
            gender="female",
            usage_counts=high_usage,
        )
        assert result == "bf_emma", (
            f"Tag match should win over usage count, got {result}"
        )

    def test_best_tag_match_wins(self):
        """The voice with more matching tags beats one with fewer tags."""
        result = match_voice("warm british female", gender="female", usage_counts={})
        # British female voices start with bf_
        assert result is not None
        assert result.startswith("bf_"), f"Expected British female, got {result}"


# ═══════════════════════════════════════════════════════════════════════════
# match_voices_for_cast() — usage_counts threading
# ═══════════════════════════════════════════════════════════════════════════

class TestMatchVoicesForCastWithUsageCounts:
    def test_5_male_cast_all_unique(self):
        """Five male speakers should each get a distinct voice."""
        cast = {name: {"gender": "male"} for name in
                ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]}
        result = match_voices_for_cast(cast, usage_counts={})
        voices = list(result.values())
        assert len(set(voices)) == 5, f"Expected 5 unique voices, got {voices}"

    def test_5_male_cast_am_adam_appears_once(self):
        """am_adam should appear exactly once in a 5-male cast, not for all."""
        cast = {name: {"gender": "male"} for name in
                ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]}
        result = match_voices_for_cast(cast, usage_counts={})
        voices = list(result.values())
        assert voices.count("am_adam") == 1, (
            f"am_adam should appear once in a 5-male cast, got {voices}"
        )

    def test_am_adam_still_in_pool(self):
        """am_adam must remain in the voice pool — it should appear for small casts."""
        cast = {"Solo": {"gender": "male"}}
        result = match_voices_for_cast(cast, usage_counts={})
        assert result["Solo"] == "am_adam", (
            f"With one male speaker and no usage, expected am_adam, got {result['Solo']}"
        )

    def test_heavy_usage_of_am_adam_routes_to_other_voice(self):
        """If am_adam is heavily used, a new solo male speaker gets a fresher voice."""
        usage = {"am_adam": 50}
        cast = {"New Guy": {"gender": "male"}}
        result = match_voices_for_cast(cast, usage_counts=usage)
        assert result["New Guy"] != "am_adam", (
            f"Expected a voice other than am_adam when it's heavily used"
        )

    def test_usage_counts_none_behaves_as_before(self):
        """Omitting usage_counts keeps legacy am_adam-first behaviour."""
        cast = {"Solo": {"gender": "male"}}
        result = match_voices_for_cast(cast, usage_counts=None)
        assert result["Solo"] == "am_adam"

    def test_usage_counts_empty_dict_behaves_like_zero_usage(self):
        """An empty usage_counts dict is equivalent to all-zero usage."""
        cast = {"Solo": {"gender": "male"}}
        r_none = match_voices_for_cast(cast, usage_counts=None)
        r_empty = match_voices_for_cast(cast, usage_counts={})
        assert r_none == r_empty

    def test_large_cast_wraps_without_crash(self):
        """A cast larger than the voice pool should wrap without KeyError."""
        # 15 males, only 11 in the Kokoro catalog — should wrap.
        cast = {f"Speaker{i}": {"gender": "male"} for i in range(15)}
        result = match_voices_for_cast(cast, usage_counts={})
        assert len(result) == 15
        assert all(v != "default" for v in result.values()), (
            f"Some speakers got 'default': {result}"
        )

    def test_mixed_gender_cast_routes_correctly(self):
        """Female characters get female voices, males get male voices."""
        cast = {
            "Alice": {"gender": "female"},
            "Bob": {"gender": "male"},
            "Carol": {"gender": "female"},
            "Dave": {"gender": "male"},
        }
        result = match_voices_for_cast(cast, usage_counts={})
        for name in ("Alice", "Carol"):
            assert result[name][:2] in ("af", "bf"), (
                f"{name} should be female, got {result[name]}"
            )
        for name in ("Bob", "Dave"):
            assert result[name][:2] in ("am", "bm"), (
                f"{name} should be male, got {result[name]}"
            )

    def test_usage_influences_order_not_correctness(self):
        """Heavy usage of one voice shifts who gets it but all speakers still
        get a valid, non-default voice."""
        usage = {v: 10 for v in ALL_MALE_KOKORO[:5]}  # first 5 males heavily used
        cast = {f"S{i}": {"gender": "male"} for i in range(5)}
        result = match_voices_for_cast(cast, usage_counts=usage)
        assert all(v in KOKORO_VOICE_CATALOG for v in result.values()), (
            f"All voices should be in the catalog: {result}"
        )
        assert len(set(result.values())) == 5, f"Should still be unique: {result}"
