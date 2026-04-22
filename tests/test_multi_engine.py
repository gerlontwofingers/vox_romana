"""Tests for the unified voice catalog and MultiEngine wrapper.

These cover the post-Phase-3 voice-pool refactor: every voice now knows
which engine owns it, the unified catalog merges Kokoro + Piper without
collision, and MultiEngine dispatches synthesize() calls based on the
descriptor's ``engine`` tag.
"""

from __future__ import annotations

import pytest

from vox_romana.tts.engine import BaseTTSEngine, SynthesizedAudio
from vox_romana.tts.engines.multi_engine import MultiEngine
from vox_romana.tts.voice_catalog import (
    KOKORO_VOICE_CATALOG,
    PIPER_VOICE_CATALOG,
    UNIFIED_VOICE_CATALOG,
)


# ── Fakes ────────────────────────────────────────────────────────────


class _FakeEngine(BaseTTSEngine):
    """Minimal engine that records which calls it received."""

    def __init__(self, name: str, voices: list[str]):
        self._name = name
        self._voices = list(voices)
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def available_voices(self) -> list[str]:
        return list(self._voices)

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "default",
        emotion: str = "",
        speed: float = 1.0,
        **kwargs,
    ) -> SynthesizedAudio:
        self.calls.append(voice)
        return SynthesizedAudio(audio=b"", sample_rate=22050)


# ── Catalog wiring ───────────────────────────────────────────────────


class TestUnifiedCatalog:
    def test_every_kokoro_voice_tagged_kokoro(self):
        assert KOKORO_VOICE_CATALOG, "Kokoro catalog should not be empty."
        for vid, desc in KOKORO_VOICE_CATALOG.items():
            assert desc.engine == "kokoro", f"{vid} not tagged kokoro"

    def test_every_piper_voice_tagged_piper(self):
        assert PIPER_VOICE_CATALOG, "Piper catalog should not be empty."
        for vid, desc in PIPER_VOICE_CATALOG.items():
            assert desc.engine == "piper", f"{vid} not tagged piper"

    def test_unified_has_no_collisions(self):
        # Sanity: union size == sum of parts (verified by no overwrite).
        expected = len(KOKORO_VOICE_CATALOG) + len(PIPER_VOICE_CATALOG)
        assert len(UNIFIED_VOICE_CATALOG) == expected

    def test_unified_contains_both_engines(self):
        engines = {desc.engine for desc in UNIFIED_VOICE_CATALOG.values()}
        assert engines == {"kokoro", "piper"}


# ── MultiEngine dispatch ─────────────────────────────────────────────


class TestMultiEngineDispatch:
    def test_routes_kokoro_voice_to_kokoro_engine(self):
        kokoro_voice = next(iter(KOKORO_VOICE_CATALOG.keys()))
        kokoro = _FakeEngine("Kokoro", [kokoro_voice])
        piper = _FakeEngine("Piper", [])
        multi = MultiEngine([kokoro, piper])

        multi.synthesize("hi", voice=kokoro_voice)

        assert kokoro.calls == [kokoro_voice]
        assert piper.calls == []

    def test_routes_piper_voice_to_piper_engine(self):
        piper_voice = next(iter(PIPER_VOICE_CATALOG.keys()))
        kokoro = _FakeEngine("Kokoro", [])
        piper = _FakeEngine("Piper", [piper_voice])
        multi = MultiEngine([kokoro, piper])

        multi.synthesize("hi", voice=piper_voice)

        assert piper.calls == [piper_voice]
        assert kokoro.calls == []

    def test_falls_back_to_engine_that_lists_voice(self):
        # Voice not in catalog → MultiEngine probes engine.available_voices().
        kokoro = _FakeEngine("Kokoro", [])
        piper = _FakeEngine("Piper", ["some-fresh-piper-model"])
        multi = MultiEngine([kokoro, piper])

        multi.synthesize("hi", voice="some-fresh-piper-model")

        assert piper.calls == ["some-fresh-piper-model"]
        assert kokoro.calls == []

    def test_unknown_voice_raises(self):
        kokoro = _FakeEngine("Kokoro", [])
        piper = _FakeEngine("Piper", [])
        multi = MultiEngine([kokoro, piper])

        with pytest.raises(ValueError, match="No wrapped engine"):
            multi.synthesize("hi", voice="ghost-voice-9000")

    def test_available_voices_unions_both(self):
        kokoro = _FakeEngine("Kokoro", ["a", "b"])
        piper = _FakeEngine("Piper", ["c", "b"])  # 'b' overlap
        multi = MultiEngine([kokoro, piper])

        voices = multi.available_voices()

        assert set(voices) == {"a", "b", "c"}
        assert len(voices) == 3  # 'b' deduplicated

    def test_empty_engine_list_rejected(self):
        with pytest.raises(ValueError):
            MultiEngine([])
