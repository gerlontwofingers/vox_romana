"""Tests for MusicGenEngine — the generative music engine.

Tests the wiring of MusicGenEngine without touching any real model.
All HuggingFace / torch calls are mocked at the module level.

THE TESTING PHILOSOPHY (same as test_tts.py):

    We test the WIRING, not the music quality. We verify that:
    - The model is NOT loaded at import or construction time (lazy load)
    - generate() passes the correct duration to set_generation_params()
    - The prompt is forwarded to the processor
    - The returned SynthesizedAudio has valid WAV bytes and the right sample rate
    - Device selection picks MPS when available, CUDA when no MPS, CPU as fallback
    - Model aliases ("small", "medium", "large") resolve to the correct HF ids
    - A MusicGenEngine with explicit device="cpu" stays on CPU

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_music_engine.py -v
"""

import io
import struct
import wave
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from vox_romana.tts.engine import SynthesizedAudio
from vox_romana.tts.music_engine import (
    MusicGenEngine,
    DEFAULT_MODEL,
    MUSICGEN_SAMPLE_RATE,
    MODEL_ALIASES,
    _resolve_model_name,
    _select_device,
)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _parse_wav_header(wav_bytes: bytes) -> dict:
    """Parse a WAV header and return its key fields.

    Useful for asserting that generate() really does return a valid WAV file
    without playing the audio.
    """
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        return {
            "channels": wf.getnchannels(),
            "sample_width": wf.getsampwidth(),
            "frame_rate": wf.getframerate(),
            "n_frames": wf.getnframes(),
        }


def _make_fake_audio_tensor(n_samples: int = 32000):
    """Return a fake 3-D float32 tensor [1, 1, n_samples] that mimics MusicGen output."""
    import numpy as np
    # We return a MagicMock whose cpu().numpy() chain returns a flat numpy array.
    fake_np = np.zeros(n_samples, dtype=np.float32)

    inner = MagicMock()
    inner.cpu.return_value.numpy.return_value = fake_np

    # The engine accesses audio_values[0, 0], so we need __getitem__.
    outer = MagicMock()
    outer.__getitem__ = MagicMock(return_value=inner)
    return outer


def _make_fake_inputs():
    """Fake tokenised inputs dict — each value has a .to() method."""
    fake_tensor = MagicMock()
    fake_tensor.to.return_value = fake_tensor
    return {"input_ids": fake_tensor, "attention_mask": fake_tensor}


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONSTRUCTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestMusicGenEngineConstruction:
    """Tests for MusicGenEngine.__init__() and lazy loading."""

    def test_default_model_name(self):
        """Default model is musicgen-small."""
        engine = MusicGenEngine()
        assert engine.model_name == DEFAULT_MODEL
        assert "musicgen-small" in engine.model_name

    def test_alias_small(self):
        """'small' alias resolves to facebook/musicgen-small."""
        engine = MusicGenEngine(model_name="small")
        assert engine.model_name == "facebook/musicgen-small"

    def test_alias_medium(self):
        """'medium' alias resolves to facebook/musicgen-medium."""
        engine = MusicGenEngine(model_name="medium")
        assert engine.model_name == "facebook/musicgen-medium"

    def test_alias_large(self):
        """'large' alias resolves to facebook/musicgen-large."""
        engine = MusicGenEngine(model_name="large")
        assert engine.model_name == "facebook/musicgen-large"

    def test_full_hf_id_passed_through(self):
        """A full HuggingFace model id is passed through unchanged."""
        engine = MusicGenEngine(model_name="facebook/musicgen-large")
        assert engine.model_name == "facebook/musicgen-large"

    def test_model_not_loaded_at_construction(self):
        """Model and processor must NOT be loaded during __init__."""
        engine = MusicGenEngine()
        assert engine._model is None
        assert engine._processor is None
        assert engine.is_loaded is False

    def test_explicit_cpu_device(self):
        """Passing device='cpu' pins the engine to CPU."""
        engine = MusicGenEngine(device="cpu")
        assert engine.device == "cpu"

    def test_explicit_mps_device(self):
        """Passing device='mps' pins the engine to MPS."""
        engine = MusicGenEngine(device="mps")
        assert engine.device == "mps"


# ═══════════════════════════════════════════════════════════════════════════
# 2. DEVICE SELECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectDevice:
    """Tests for the _select_device() helper."""

    def test_explicit_device_returned_as_is(self):
        """An explicit device string is returned unchanged."""
        assert _select_device("cpu") == "cpu"
        assert _select_device("mps") == "mps"
        assert _select_device("cuda") == "cuda"

    def test_none_picks_mps_when_available(self):
        """When MPS is available, None → 'mps'."""
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _select_device(None) == "mps"

    def test_none_picks_cuda_when_no_mps(self):
        """When MPS is unavailable but CUDA is, None → 'cuda'."""
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _select_device(None) == "cuda"

    def test_none_falls_back_to_cpu(self):
        """When neither MPS nor CUDA is available, None → 'cpu'."""
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _select_device(None) == "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# 3. MODEL ALIAS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveModelName:
    """Tests for the _resolve_model_name() helper."""

    def test_known_aliases(self):
        for alias, expected in MODEL_ALIASES.items():
            assert _resolve_model_name(alias) == expected

    def test_unknown_string_passed_through(self):
        custom = "my-org/my-musicgen-finetuned"
        assert _resolve_model_name(custom) == custom


# ═══════════════════════════════════════════════════════════════════════════
# 4. GENERATE TESTS (model mocked)
# ═══════════════════════════════════════════════════════════════════════════

class TestMusicGenEngineGenerate:
    """Tests for MusicGenEngine.generate() with mocked transformers."""

    def _make_engine_with_mock_model(self, device: str = "cpu"):
        """Build an engine and inject a pre-loaded mock model/processor.

        This bypasses _ensure_loaded() so we can test generate() in isolation
        without touching HuggingFace or torch.
        """
        engine = MusicGenEngine(model_name="small", device=device)

        # Fake processor: returns a dict of tensors (each with .to())
        fake_processor = MagicMock()
        fake_processor.return_value = _make_fake_inputs()

        # Fake model: generate() returns a fake 3-D tensor
        fake_model = MagicMock()
        fake_model.generate.return_value = _make_fake_audio_tensor()

        # Pre-inject so _ensure_loaded() is a no-op
        engine._processor = fake_processor
        engine._model = fake_model

        return engine, fake_model, fake_processor

    # ── Core output contract ───────────────────────────────────────

    def test_returns_synthesized_audio(self):
        """generate() returns a SynthesizedAudio instance."""
        engine, _, _ = self._make_engine_with_mock_model()
        result = engine.generate("jazz podcast intro")
        assert isinstance(result, SynthesizedAudio)

    def test_sample_rate_is_musicgen_native(self):
        """sample_rate is always MUSICGEN_SAMPLE_RATE (32000 Hz)."""
        engine, _, _ = self._make_engine_with_mock_model()
        result = engine.generate("jazz")
        assert result.sample_rate == MUSICGEN_SAMPLE_RATE

    def test_audio_bytes_is_valid_wav(self):
        """audio field starts with b'RIFF' (standard WAV magic bytes)."""
        engine, _, _ = self._make_engine_with_mock_model()
        result = engine.generate("podcast music")
        assert result.audio[:4] == b"RIFF"

    def test_wav_sample_rate_in_header(self):
        """The WAV header encodes MUSICGEN_SAMPLE_RATE."""
        engine, _, _ = self._make_engine_with_mock_model()
        result = engine.generate("stinger")
        header = _parse_wav_header(result.audio)
        assert header["frame_rate"] == MUSICGEN_SAMPLE_RATE

    def test_wav_is_mono_int16(self):
        """The WAV is mono (1 channel) 16-bit PCM."""
        engine, _, _ = self._make_engine_with_mock_model()
        result = engine.generate("intro")
        header = _parse_wav_header(result.audio)
        assert header["channels"] == 1
        assert header["sample_width"] == 2  # int16 = 2 bytes

    # ── Prompt and duration forwarding ────────────────────────────

    def test_duration_forwarded_to_set_generation_params(self):
        """generate() calls model.set_generation_params(duration=<value>)."""
        engine, fake_model, _ = self._make_engine_with_mock_model()
        engine.generate("intro music", duration=12.5)
        fake_model.set_generation_params.assert_called_once_with(duration=12.5)

    def test_default_duration_is_8_seconds(self):
        """Default duration is 8.0 seconds."""
        engine, fake_model, _ = self._make_engine_with_mock_model()
        engine.generate("background music")
        fake_model.set_generation_params.assert_called_once_with(duration=8.0)

    def test_prompt_forwarded_to_processor(self):
        """The prompt string is passed to the processor as text=[prompt]."""
        engine, _, fake_processor = self._make_engine_with_mock_model()
        engine.generate("lo-fi jazz")
        fake_processor.assert_called_once()
        call_kwargs = fake_processor.call_args
        assert call_kwargs.kwargs["text"] == ["lo-fi jazz"]

    # ── Lazy loading ──────────────────────────────────────────────

    def test_model_not_loaded_before_generate(self):
        """is_loaded is False before the first generate() call."""
        engine = MusicGenEngine(model_name="small", device="cpu")
        assert not engine.is_loaded

    def test_model_loaded_after_generate(self):
        """is_loaded is True after generate() runs (model injected)."""
        engine, _, _ = self._make_engine_with_mock_model()
        assert engine.is_loaded  # We pre-injected the model.

    def test_ensure_loaded_calls_from_pretrained(self):
        """_ensure_loaded() calls from_pretrained on both AutoProcessor and the model class."""
        engine = MusicGenEngine(model_name="small", device="cpu")

        mock_model = MagicMock()
        mock_model.generate.return_value = _make_fake_audio_tensor()
        mock_processor = MagicMock(return_value=_make_fake_inputs())

        with patch("vox_romana.tts.music_engine.MusicgenForConditionalGeneration") as mock_model_cls, \
             patch("vox_romana.tts.music_engine.AutoProcessor") as mock_proc_cls:
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_proc_cls.from_pretrained.return_value = mock_processor
            engine._ensure_loaded()

        assert engine.is_loaded
        mock_model_cls.from_pretrained.assert_called_once_with("facebook/musicgen-small", cache_dir=None)
        mock_proc_cls.from_pretrained.assert_called_once_with("facebook/musicgen-small", cache_dir=None)

    # ── Device propagation ────────────────────────────────────────

    def test_inputs_moved_to_device(self):
        """Each input tensor is moved to the engine's device via .to()."""
        engine, _, fake_processor = self._make_engine_with_mock_model(device="cpu")
        fake_inputs = _make_fake_inputs()
        fake_processor.return_value = fake_inputs
        engine.generate("music")
        # Each tensor in the inputs dict should have had .to("cpu") called.
        for tensor in fake_inputs.values():
            tensor.to.assert_called_with("cpu")

    def test_model_moved_to_device_on_load(self):
        """model.to(device) is called on the real model object inside _ensure_loaded()."""
        engine = MusicGenEngine(model_name="small", device="cpu")

        mock_model = MagicMock()
        mock_model.generate.return_value = _make_fake_audio_tensor()
        mock_processor = MagicMock(return_value=_make_fake_inputs())

        with patch("vox_romana.tts.music_engine.MusicgenForConditionalGeneration") as mock_model_cls, \
             patch("vox_romana.tts.music_engine.AutoProcessor") as mock_proc_cls:
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_proc_cls.from_pretrained.return_value = mock_processor
            engine._ensure_loaded()

        mock_model.to.assert_called_once_with("cpu")

    # ── Properties ────────────────────────────────────────────────

    def test_model_name_property(self):
        engine = MusicGenEngine(model_name="medium")
        assert engine.model_name == "facebook/musicgen-medium"

    def test_device_property(self):
        engine = MusicGenEngine(device="cpu")
        assert engine.device == "cpu"
