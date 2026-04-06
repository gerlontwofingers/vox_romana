"""vox_romana.tts — the text-to-speech pipeline.

This package turns debate JSON scripts into spoken audio files.
Think of it as the "sound department" of a film studio:

    ┌─────────────────────────────────────────────────────────┐
    │  Debate JSON  →  Renderer  →  Engine  →  Mixer  →  MP3 │
    └─────────────────────────────────────────────────────────┘

COMPONENTS:
    engine.py     — BaseTTSEngine ABC, SynthesizedAudio, VoiceConfig
    mixer.py      — AudioMixer (stitches clips with pauses & overlaps)
    renderer.py   — TTSRenderer (orchestrates engine + mixer)
    engines/      — Plug-in engine implementations:
        piper_engine.py  — Local, free, CPU-based (ONNX)
        kokoro_engine.py — Local, free, GPU-based (MLX, Apple Silicon)

QUICK START:
    from vox_romana.tts import TTSRenderer
    from vox_romana.tts.engines.kokoro_engine import KokoroEngine

    engine = KokoroEngine()
    renderer = TTSRenderer(engine)
    result = renderer.render_file("output/debate.json", "output/debate.mp3")
    print(f"Saved {result.duration_ms}ms of audio to {result.output_path}")
"""

from .engine import BaseTTSEngine, SynthesizedAudio, VoiceConfig
from .mixer import AudioMixer, AudioClip, PAUSE_MAP, DEFAULT_PAUSE_MS
from .renderer import TTSRenderer, RenderResult

__all__ = [
    # Engine interface
    "BaseTTSEngine",
    "SynthesizedAudio",
    "VoiceConfig",
    # Mixer
    "AudioMixer",
    "AudioClip",
    "PAUSE_MAP",
    "DEFAULT_PAUSE_MS",
    # Renderer
    "TTSRenderer",
    "RenderResult",
]
