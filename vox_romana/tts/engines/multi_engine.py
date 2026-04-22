"""MultiEngine — a TTS engine that dispatches to multiple backends.

When you want voice variety beyond what a single engine can provide,
load both Piper and Kokoro and let the renderer choose voices from the
combined pool. MultiEngine wraps both and routes each ``synthesize()``
call to the engine that owns the requested voice_id.

THE TRANSLATOR ANALOGY:

    A multilingual translator at a UN summit doesn't speak French and
    Mandarin at the same time. They listen to the request, recognise
    which language is needed, and respond in that language. MultiEngine
    works the same way: a single conversation partner for the renderer,
    but underneath it dispatches to whichever engine actually owns the
    voice that was asked for.

USAGE:

    from vox_romana.tts.engines.multi_engine import MultiEngine

    engine = MultiEngine.from_engine_names(["kokoro", "piper"])
    renderer = TTSRenderer(engine)
    renderer.render_file("show.json", "show.mp3")

    # Or pass instantiated engines directly:
    engine = MultiEngine([KokoroEngine(), PiperEngine()])

DISPATCH RULES:

    1. Look up the voice in UNIFIED_VOICE_CATALOG → get .engine field.
    2. Find the wrapped engine matching that name → forward the call.
    3. If the voice isn't in the catalog, fall back to whichever engine
       reports it in available_voices(). If both do, the first one wins.
    4. If no engine knows the voice, raise ValueError with diagnostic.
"""

from __future__ import annotations

from ..engine import BaseTTSEngine, SynthesizedAudio
from ..voice_catalog import UNIFIED_VOICE_CATALOG


class MultiEngine(BaseTTSEngine):
    """Routes synthesize() calls to one of several wrapped TTS engines.

    Attributes:
        _engines:    Tuple of wrapped BaseTTSEngine instances.
        _by_name:    Lower-cased engine name → engine instance.
        _voice_owner_cache: voice_id → engine instance, populated lazily.
    """

    def __init__(self, engines: list[BaseTTSEngine]) -> None:
        if not engines:
            raise ValueError("MultiEngine needs at least one wrapped engine.")
        self._engines: tuple[BaseTTSEngine, ...] = tuple(engines)
        self._by_name: dict[str, BaseTTSEngine] = {
            e.name.lower(): e for e in self._engines
        }
        self._voice_owner_cache: dict[str, BaseTTSEngine] = {}

    @classmethod
    def from_engine_names(cls, names: list[str]) -> "MultiEngine":
        """Construct from a list of engine names.

        Lazily imports each engine so users without a backend installed
        only pay the import cost for the engines they actually request.
        """
        instances: list[BaseTTSEngine] = []
        seen: set[str] = set()
        for name in names:
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            if key == "kokoro":
                from .kokoro_engine import KokoroEngine
                instances.append(KokoroEngine())
            elif key == "piper":
                from .piper_engine import PiperEngine
                instances.append(PiperEngine())
            else:
                raise ValueError(f"Unknown engine name: {name!r}")
        return cls(instances)

    # ── BaseTTSEngine interface ──────────────────────────────────────

    @property
    def name(self) -> str:
        return "Multi(" + "+".join(e.name for e in self._engines) + ")"

    def available_voices(self) -> list[str]:
        """Return the union of voices from all wrapped engines.

        Voices that exist in both engines' lists appear once (first
        engine wins for ordering). The renderer uses this list to
        decide which voices it can assign.
        """
        out: list[str] = []
        seen: set[str] = set()
        for engine in self._engines:
            for vid in engine.available_voices():
                if vid not in seen:
                    seen.add(vid)
                    out.append(vid)
        return out

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "default",
        emotion: str = "",
        speed: float = 1.0,
        **kwargs,
    ) -> SynthesizedAudio:
        """Dispatch to whichever wrapped engine owns the requested voice."""
        engine = self._engine_for_voice(voice)
        return engine.synthesize(
            text, voice=voice, emotion=emotion, speed=speed, **kwargs,
        )

    # ── Dispatch ─────────────────────────────────────────────────────

    def _engine_for_voice(self, voice_id: str) -> BaseTTSEngine:
        """Return the engine that should handle this voice_id."""
        cached = self._voice_owner_cache.get(voice_id)
        if cached is not None:
            return cached

        # Prefer the catalog: it's the source of truth for voice→engine
        # mapping. Falls back to per-engine probing for voices that
        # aren't catalogued (e.g. a freshly added Piper model).
        descriptor = UNIFIED_VOICE_CATALOG.get(voice_id)
        if descriptor is not None:
            engine = self._by_name.get(descriptor.engine.lower())
            if engine is not None:
                self._voice_owner_cache[voice_id] = engine
                return engine

        for engine in self._engines:
            try:
                voices = engine.available_voices()
            except Exception:  # noqa: BLE001 — engine probe should never crash dispatch
                continue
            if voice_id in voices:
                self._voice_owner_cache[voice_id] = engine
                return engine

        raise ValueError(
            f"No wrapped engine knows voice {voice_id!r}. "
            f"Available engines: {[e.name for e in self._engines]}"
        )
