"""Tests for VoiceRegistry — character locking and usage tracking.

Covers:
    1. series_key_for() — priority logic
    2. resolve() — miss returns None, hit returns voice_id
    3. register() — first call locks, subsequent calls do NOT overwrite
    4. series isolation — same character, different series_key → independent rows
    5. record_usage() — upsert semantics, increments accumulate
    6. usage_counts() — empty dict when no records, correct counts after inserts
    7. context-manager protocol (__enter__/__exit__)
    8. All tests use ":memory:" — no disk I/O, no shared state
"""

import pytest

from vox_romana.tts.voice_registry import VoiceRegistry, series_key_for


# ═══════════════════════════════════════════════════════════════════════════
# series_key_for()
# ═══════════════════════════════════════════════════════════════════════════

class TestSeriesKeyFor:
    def test_series_id_takes_priority_over_show_name(self):
        assert series_key_for("my-series", "My Show") == "my-series"

    def test_show_name_used_when_no_series_id(self):
        assert series_key_for(None, "My Show") == "My Show"

    def test_global_fallback_when_both_none(self):
        assert series_key_for(None, None) == "global"

    def test_empty_string_series_id_falls_through_to_show_name(self):
        # Empty string is falsy — should fall through to show_name.
        assert series_key_for("", "My Show") == "My Show"

    def test_empty_string_show_name_falls_through_to_global(self):
        assert series_key_for(None, "") == "global"

    def test_both_empty_returns_global(self):
        assert series_key_for("", "") == "global"


# ═══════════════════════════════════════════════════════════════════════════
# VoiceRegistry — resolve / register
# ═══════════════════════════════════════════════════════════════════════════

class TestResolve:
    def test_unknown_character_returns_none(self):
        reg = VoiceRegistry(db_path=":memory:")
        assert reg.resolve("Dr. Chen", "Science Hour") is None

    def test_unknown_series_returns_none(self):
        reg = VoiceRegistry(db_path=":memory:")
        reg.register("Dr. Chen", "af_jessica", "Science Hour")
        assert reg.resolve("Dr. Chen", "Other Show") is None

    def test_round_trip_register_then_resolve(self):
        reg = VoiceRegistry(db_path=":memory:")
        reg.register("Dr. Chen", "af_jessica", "Science Hour")
        assert reg.resolve("Dr. Chen", "Science Hour") == "af_jessica"

    def test_resolve_returns_exact_voice_id(self):
        reg = VoiceRegistry(db_path=":memory:")
        reg.register("Host", "bm_george", "weekly-tech")
        assert reg.resolve("Host", "weekly-tech") == "bm_george"


class TestRegister:
    def test_first_register_locks_voice(self):
        reg = VoiceRegistry(db_path=":memory:")
        reg.register("Alice", "am_adam", "show-a")
        assert reg.resolve("Alice", "show-a") == "am_adam"

    def test_second_register_does_not_overwrite_voice(self):
        reg = VoiceRegistry(db_path=":memory:")
        reg.register("Alice", "am_adam", "show-a")
        reg.register("Alice", "am_echo", "show-a")   # should be ignored
        assert reg.resolve("Alice", "show-a") == "am_adam"

    def test_register_is_idempotent_same_voice(self):
        reg = VoiceRegistry(db_path=":memory:")
        reg.register("Alice", "am_adam", "show-a")
        reg.register("Alice", "am_adam", "show-a")
        assert reg.resolve("Alice", "show-a") == "am_adam"

    def test_different_series_get_independent_locks(self):
        reg = VoiceRegistry(db_path=":memory:")
        reg.register("Alice", "am_adam", "show-a")
        reg.register("Alice", "am_echo", "show-b")
        assert reg.resolve("Alice", "show-a") == "am_adam"
        assert reg.resolve("Alice", "show-b") == "am_echo"

    def test_different_characters_same_series(self):
        reg = VoiceRegistry(db_path=":memory:")
        reg.register("Alice", "af_heart", "my-podcast")
        reg.register("Bob", "am_michael", "my-podcast")
        assert reg.resolve("Alice", "my-podcast") == "af_heart"
        assert reg.resolve("Bob", "my-podcast") == "am_michael"

    def test_register_many_characters(self):
        reg = VoiceRegistry(db_path=":memory:")
        characters = [
            ("Host", "af_heart"),
            ("Guest1", "am_echo"),
            ("Guest2", "bm_george"),
            ("Narrator", "bf_emma"),
        ]
        for name, voice in characters:
            reg.register(name, voice, "multi-cast")
        for name, voice in characters:
            assert reg.resolve(name, "multi-cast") == voice


# ═══════════════════════════════════════════════════════════════════════════
# VoiceRegistry — record_usage / usage_counts
# ═══════════════════════════════════════════════════════════════════════════

class TestUsageCounts:
    def test_empty_when_no_usage_recorded(self):
        reg = VoiceRegistry(db_path=":memory:")
        assert reg.usage_counts() == {}

    def test_single_record_usage(self):
        reg = VoiceRegistry(db_path=":memory:")
        reg.record_usage("am_echo")
        assert reg.usage_counts() == {"am_echo": 1}

    def test_multiple_calls_accumulate(self):
        reg = VoiceRegistry(db_path=":memory:")
        reg.record_usage("af_jessica")
        reg.record_usage("af_jessica")
        reg.record_usage("af_jessica")
        counts = reg.usage_counts()
        assert counts["af_jessica"] == 3

    def test_multiple_voices_tracked_independently(self):
        reg = VoiceRegistry(db_path=":memory:")
        reg.record_usage("af_jessica")
        reg.record_usage("af_jessica")
        reg.record_usage("am_echo")
        counts = reg.usage_counts()
        assert counts["af_jessica"] == 2
        assert counts["am_echo"] == 1

    def test_unused_voices_absent_from_counts(self):
        reg = VoiceRegistry(db_path=":memory:")
        reg.record_usage("af_heart")
        counts = reg.usage_counts()
        assert "am_adam" not in counts
        assert "af_heart" in counts

    def test_register_does_not_affect_usage_counts(self):
        # register() locks character → voice; does NOT increment usage.
        reg = VoiceRegistry(db_path=":memory:")
        reg.register("Alice", "af_heart", "my-show")
        assert reg.usage_counts() == {}

    def test_record_usage_upserts_new_voice(self):
        reg = VoiceRegistry(db_path=":memory:")
        # First call creates the row.
        reg.record_usage("bm_daniel")
        assert reg.usage_counts()["bm_daniel"] == 1
        # Second call increments it.
        reg.record_usage("bm_daniel")
        assert reg.usage_counts()["bm_daniel"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# VoiceRegistry — context manager
# ═══════════════════════════════════════════════════════════════════════════

class TestContextManager:
    def test_context_manager_returns_self(self):
        with VoiceRegistry(db_path=":memory:") as reg:
            assert isinstance(reg, VoiceRegistry)

    def test_operations_work_inside_with_block(self):
        with VoiceRegistry(db_path=":memory:") as reg:
            reg.register("Alice", "af_heart", "test-show")
            reg.record_usage("af_heart")
            assert reg.resolve("Alice", "test-show") == "af_heart"
            assert reg.usage_counts() == {"af_heart": 1}

    def test_close_is_callable(self):
        reg = VoiceRegistry(db_path=":memory:")
        reg.close()  # Should not raise.


# ═══════════════════════════════════════════════════════════════════════════
# VoiceRegistry — schema idempotency
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemaIdempotency:
    def test_multiple_registry_instances_on_same_memory_db(self):
        # Each :memory: connection gets its own DB — this just checks
        # that _ensure_schema() doesn't raise on a fresh connection.
        reg1 = VoiceRegistry(db_path=":memory:")
        reg2 = VoiceRegistry(db_path=":memory:")
        reg1.register("Alice", "af_heart", "show-a")
        reg2.register("Alice", "am_adam", "show-a")
        # They're independent.
        assert reg1.resolve("Alice", "show-a") == "af_heart"
        assert reg2.resolve("Alice", "show-a") == "am_adam"
        reg1.close()
        reg2.close()

    def test_ensure_schema_safe_on_existing_tables(self, tmp_path):
        db_file = tmp_path / "test.db"
        # First open creates the schema.
        reg1 = VoiceRegistry(db_path=str(db_file))
        reg1.register("Host", "af_heart", "my-show")
        reg1.close()
        # Second open must not fail or wipe existing data.
        reg2 = VoiceRegistry(db_path=str(db_file))
        assert reg2.resolve("Host", "my-show") == "af_heart"
        reg2.close()

    def test_data_persists_across_reopens(self, tmp_path):
        db_file = tmp_path / "persist.db"

        with VoiceRegistry(db_path=str(db_file)) as reg:
            reg.register("Alice", "af_jessica", "show-x")
            reg.record_usage("af_jessica")
            reg.record_usage("af_jessica")

        with VoiceRegistry(db_path=str(db_file)) as reg:
            assert reg.resolve("Alice", "show-x") == "af_jessica"
            assert reg.usage_counts()["af_jessica"] == 2
