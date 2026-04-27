# Video mode setup guide

`--mode video` extends the normal generate + render pipeline with a third
phase that produces an MP4.  The animation phase runs entirely locally —
no external API is needed.

```
prompt → script (JSON) → audio (MP3/WAV) → frames (Stable Diffusion) → video (MP4)
```

---

## 1. Python packages

Install the animation extras on top of the base requirements:

```bash
pip install torch diffusers accelerate peft moviepy Pillow
```

| Package | Purpose |
|---------|---------|
| `torch` | Tensor runtime (MPS on Apple Silicon, CUDA on NVIDIA, CPU fallback) |
| `diffusers ≥0.32` | Stable Diffusion pipeline (SDXL and SD 1.5) |
| `accelerate` | Accelerated model loading |
| `peft` | Required by some diffusers checkpoints |
| `moviepy ≥1.0.3,<2.0` | Ken Burns compositing + MP4 export |
| `Pillow` | Image I/O |

Verify after installing:

```bash
python preflight.py
```

You should see `✓` next to `diffusers`, `moviepy`, and `Pillow`.

---

## 2. System packages

### FFmpeg

MoviePy requires `ffmpeg` for video export.  The base README already
covers this, but confirm it is on your PATH:

```bash
ffmpeg -version
```

### ImageMagick (optional)

Text overlays (lower-thirds / speaker labels) use MoviePy's `TextClip`,
which requires ImageMagick.  The video renders without it — text layers
are silently skipped.

```bash
# macOS
brew install imagemagick

# Ubuntu / Debian
sudo apt install imagemagick
```

`preflight.py` will show `⚠ imagemagick — not found` when it is absent
and `✓ imagemagick` once installed.

---

## 3. Hardware notes

### Apple Silicon (M1 / M2 / M3)

vox_romana automatically selects `device = mps` on Apple Silicon.

| Platform | Model | VRAM estimate | Typical time (M1 Max 32 GB) |
|----------|-------|---------------|-----------------------------|
| `fast` | SD 1.5 — 512×512 | ~2 GB | ~8–15 s / frame |
| `youtube` | SDXL — 1920×1080 | ~8–12 GB | ~60–90 s / frame |
| `tiktok` | SDXL — 1080×1920 | ~8–12 GB | ~60–90 s / frame |
| `both` | SDXL — both sizes | ~8–12 GB | same (one SD pass, two crops) |

**MPS-specific fixes already applied:**
- SDXL VAE upcast to `float32` (prevents NaN decode on MPS).
- SD 1.5 VAE kept in `float16` (prevents dtype mismatch on MPS).
- SD 1.5 safety checker disabled (fires false positives at low step counts,
  producing all-black frames).

### NVIDIA (CUDA)

Device selection falls through to `cuda` when MPS is unavailable.  No
special configuration needed; the same model IDs work on CUDA.

### CPU fallback

All platforms work on CPU — expect 5–20× slower frame generation.

---

## 4. Model downloads

Models are downloaded to `HF_CACHE_DIR` (default: `~/.cache/huggingface`
or the path set in `vox_romana.yaml`).

| Platform | Model ID | Download size |
|----------|----------|---------------|
| `youtube`, `tiktok`, `both` | `stabilityai/stable-diffusion-xl-base-1.0` | ~6.5 GB |
| `fast` | `runwayml/stable-diffusion-v1-5` | ~4 GB |

Downloads happen automatically on first use.  To pre-cache manually:

```python
from diffusers import DiffusionPipeline
DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
```

Set a custom cache location in `vox_romana.yaml`:

```yaml
hf_cache_dir: /path/to/your/model/cache
```

---

## 5. Quick start

```bash
# Fastest possible test (SD 1.5, 512×512, 4 steps):
python run.py prompts/my_brief.txt --mode video --video-platform fast

# Full quality YouTube video:
python run.py prompts/my_brief.txt --mode video --video-platform youtube

# Custom visual style:
python run.py prompts/my_brief.txt --mode video \
    --visual-brief "cinematic documentary, warm golden hour lighting" \
    --video-platform youtube

# Fixed output stem → output/.../video/my_show_youtube.mp4
python run.py prompts/my_brief.txt --mode video \
    --name my_show --video-platform youtube
```

---

## 6. Known limitations

| Issue | Status |
|-------|--------|
| **Text overlays require ImageMagick** | Gracefully skipped when absent — video renders without lower-thirds |
| **SD 1.5 safety checker** | Fixed — disabled automatically to prevent all-black frames at low step counts |
| **Ken Burns pan (moviepy 1.x)** | Fixed — `fl()` numpy slice replaces the broken `crop(x1=lambda)` call |
| **SDXL VAE NaN on MPS** | Fixed — VAE upcast to `float32` for SDXL on Apple Silicon |
| **Bus error / exit 138–139 on macOS** | Known MPS memory pressure crash during SD inference on very long shows; reduce `--segments` or use `--video-platform fast` to diagnose |

---

## 7. Checking your setup

Run `preflight.py` to see the full status:

```bash
python preflight.py
```

A working video setup looks like:

```
  ✓  ffmpeg                            ffmpeg version 7.x ...
  ✓  diffusers                         importable
  ✓  moviepy                           importable
  ✓  Pillow                            importable
  ⚠  imagemagick                       not found — text overlays will be skipped (brew install imagemagick)
  ✓  torch                             importable
```

All four Python packages (`diffusers`, `moviepy`, `Pillow`, `torch`) must
be `✓` for the animation pipeline to run.  `imagemagick` is optional.
