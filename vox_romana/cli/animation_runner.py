"""Animation mode runner: generate → render → animate (the third phase).

Wires the ``vox_romana.animation`` pipeline into the master CLI as the
final phase of ``--mode video``.  By the time ``run_animation`` is
called, ``run.py`` has already produced both the show JSON and the
rendered audio file.  This module's job is to take those two artifacts,
plus the original brief, and produce a video.

The pipeline has three concrete stages, all encapsulated here:

    1. Hardware scan       — ``HardwareScanner.scan()``
    2. Bootstrap config    — ``AnimationBootstrap.from_brief(...)``
    3. Director (1 LLM)    — ``AnimationDirector.run(brief, events, ...)``
    4. Renderer            — ``AnimationRenderer.render(plan, ...)``

The director's LLM call must run AFTER TTS rendering because future
work (timeline sync) needs per-event audio durations from the renderer.
For v1 we pass ``event_durations=None`` and let scenes use the renderer's
built-in fallback duration; precise sync is a follow-up shot.

Why a separate module rather than inline in run.py?  Same reason as
``cli/analysis.py`` and ``cli/emotion_flow.py``: keeps the CLI dispatch
file thin and makes the runner unit-testable without driving argparse.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def run_animation(
    args: argparse.Namespace,
    json_path: Path,
    audio_path: Path,
) -> Path | None:
    """Render a video for a freshly generated show.

    Reads the brief from ``args.prompt``, loads events from the show
    JSON at ``json_path``, scans hardware, bootstraps an
    :class:`AnimationConfig`, runs the :class:`AnimationDirector` to
    produce an :class:`AnimationPlan`, then renders the video via
    :class:`AnimationRenderer`.

    Args:
        args:       Parsed namespace from run.py's parse_args().  Must
                    expose ``prompt`` (Path to the brief), ``visual_brief``
                    (str | None), ``video_output`` (Path | None), and
                    ``quiet`` (bool).
        json_path:  Path to the show JSON produced by ``--mode generate``.
                    Only the top-level ``"events"`` list is consumed.
        audio_path: Path to the rendered audio file produced by
                    ``--mode render``.  Passed to the video backend so
                    the final mux has a soundtrack.

    Returns:
        Path to the rendered video file, or ``None`` if animation was
        disabled by the bootstrap (e.g. the hardware scan decided this
        machine cannot run the pipeline).
    """
    # Imports are deferred so that ``--mode generate`` and ``--mode
    # render`` users never pay the cost of loading torch / diffusers /
    # moviepy.  Animation deps are heavy and optional.
    from vox_romana.animation import (
        AnimationBootstrap,
        AnimationDirector,
        AnimationRenderer,
        HardwareScanner,
    )
    from vox_romana.core.client import DeepSeekClient

    quiet = bool(getattr(args, "quiet", False))

    # ── Read the brief ──────────────────────────────────────────────
    # The show JSON does NOT carry the original brief — it's discarded
    # after generation.  We re-read the prompt file the user passed on
    # the CLI.  Animation needs the brief because the director uses it
    # to drive scene composition and visual style.
    prompt_path = getattr(args, "prompt", None)
    if prompt_path is None:
        print(
            "Error: --mode video requires a prompt file (positional "
            "argument).  The brief is needed to drive the animation "
            "director.",
            file=sys.stderr,
        )
        sys.exit(1)
    brief = Path(prompt_path).read_text(encoding="utf-8")

    # ── Load events from the show JSON ──────────────────────────────
    # AnimationDirector.run() accepts ``Iterable[AudioEvent | dict]`` so
    # we can pass the raw dicts straight through without rehydrating
    # them into AudioEvent dataclasses.
    with Path(json_path).open("r", encoding="utf-8") as fh:
        show_data = json.load(fh)
    events = show_data.get("events", [])
    if not events:
        print(
            f"Error: show JSON at {json_path} has no 'events' — "
            "nothing to animate.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Build the LLM client (one for both bootstrap + director) ────
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    client = DeepSeekClient(api_key=api_key)

    # ── Stage 1: hardware scan ──────────────────────────────────────
    if not quiet:
        print("\n  Scanning hardware for animation backend…")
    profile = HardwareScanner.scan()
    if not quiet:
        print(
            f"  Hardware: device={profile.device}, "
            f"vram={profile.vram_gb}GB, "
            f"recommended={profile.recommended_sd_model}"
        )

    # ── Stage 2: bootstrap (no LLM) ─────────────────────────────────
    visual_brief = getattr(args, "visual_brief", None)
    bootstrap = AnimationBootstrap(client=client)
    config = bootstrap.from_brief(
        brief,
        hardware_profile=profile,
        visual_brief=visual_brief,
    )

    if not config.enabled:
        # The bootstrap can disable animation when overrides set
        # ``enabled=False``.  We honor that and bail before any LLM or
        # render cost.  Hardware-driven disable would also surface here
        # if added in a future bootstrap revision.
        print(
            "Animation disabled by bootstrap config — skipping video "
            "render.  Audio + JSON are already on disk.",
            file=sys.stderr,
        )
        return None

    # The director needs to know which render backends the renderer
    # actually supports so it can pick a viable one.  For v1 we trust
    # the bootstrap's choice; if it's None we fall back to moviepy
    # (the only backend guaranteed to be present).
    available_backends: tuple[str, ...]
    if config.render_backend:
        available_backends = (config.render_backend,)
    else:
        available_backends = ("moviepy",)

    # ── Stage 3: director (1 LLM call) ──────────────────────────────
    if not quiet:
        print("\n  Running AnimationDirector (1 LLM call)…")
    director = AnimationDirector(client)
    plan = director.run(
        brief,
        events,
        hardware_profile=profile,
        visual_brief=visual_brief,
        available_backends=available_backends,
    )
    if not quiet:
        print(f"  Plan: {len(plan.scenes)} scenes, style={plan.style}")

    # ── Stage 4: render ─────────────────────────────────────────────
    # Resolve output directory and optional explicit filename.
    # --video-output accepts either a directory OR a full file path
    # (detected by a known video extension).
    video_output = getattr(args, "video_output", None)
    platform = getattr(args, "video_platform", "youtube")
    name_stem = getattr(args, "name", None) or None  # normalise empty str

    explicit_filename: str | None = None
    if video_output is not None:
        p = Path(video_output)
        if p.suffix.lower() in (".mp4", ".webm", ".mov"):
            output_dir = p.parent
            explicit_filename = p.name
        else:
            output_dir = p
    else:
        output_dir = audio_path.parent / "video"

    # --name overrides the stem when no explicit file path was given.
    if name_stem and not explicit_filename:
        explicit_filename = f"{name_stem}.mp4"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"\n  Platform: {platform}")
        print(f"  Rendering video to {output_dir}…")

    # ── Platform dispatch ────────────────────────────────────────────
    # SD image generation is controlled by plan.image_model (model ID)
    # and plan.resolution (width × height).  Both are mutable so we
    # can override them in place before calling the renderer.
    #
    # renderer.render() accepts output_resolution to override compositing
    # dimensions independently of image gen dimensions (used for "both"),
    # and output_filename to control the output file stem.

    if platform == "fast":
        # Speed-optimised diagnostic mode: SD 1.5 at 512×512, 4 steps,
        # 15 fps.  Swaps out the default SDXL pipeline entirely.
        # SD 1.5 does NOT need the float32 VAE upcast — only SDXL on MPS
        # triggers the NaN / black-frame bug.
        from vox_romana.animation.image_gen import ImageGenerator  # noqa: PLC0415

        _FAST_MODEL = "runwayml/stable-diffusion-v1-5"
        plan.resolution = (512, 512)
        plan.fps = 15
        plan.image_model = _FAST_MODEL

        renderer = AnimationRenderer(
            image_generator=ImageGenerator(steps=4)
        )
        fname = explicit_filename or "video_fast.mp4"
        video_path = renderer.render(
            plan,
            output_dir,
            audio_path=audio_path,
            event_durations=None,
            output_filename=fname,
        )

    elif platform == "both":
        # Generate SDXL frames ONCE at 1024×1024 (square), then composite
        # to landscape (YouTube) and portrait (TikTok) in two moviepy passes.
        # The second pass uses skip_image_gen=True — no re-diffusion.
        plan.resolution = (1024, 1024)

        yt_fname = f"{name_stem}_youtube.mp4" if name_stem else "video_youtube.mp4"
        tt_fname = f"{name_stem}_tiktok.mp4"  if name_stem else "video_tiktok.mp4"

        renderer = AnimationRenderer()

        # Pass 1: generate frames + composite to 1920×1080
        renderer.render(
            plan,
            output_dir,
            audio_path=audio_path,
            event_durations=None,
            output_resolution=(1920, 1080),
            output_filename=yt_fname,
        )
        # Pass 2: reuse frames, composite to 1080×1920
        renderer.render(
            plan,
            output_dir,
            audio_path=audio_path,
            event_durations=None,
            skip_image_gen=True,
            output_resolution=(1080, 1920),
            output_filename=tt_fname,
        )
        # Primary return value is the YouTube file.
        video_path = output_dir / yt_fname

    else:
        # youtube or tiktok: single render pass at the target resolution.
        _RESOLUTIONS = {
            "youtube": (1920, 1080),
            "tiktok":  (1080, 1920),
        }
        plan.resolution = _RESOLUTIONS[platform]
        fname = explicit_filename or f"video_{platform}.mp4"

        renderer = AnimationRenderer()
        video_path = renderer.render(
            plan,
            output_dir,
            audio_path=audio_path,
            event_durations=None,
            output_filename=fname,
        )

    # Flush MPS command queue before returning so that torch/diffusers
    # tensors are released while the interpreter is still in a valid
    # state.  Without this, the MPS finalizer runs during interpreter
    # shutdown and triggers a SIGABRT (exit 139).
    try:
        import torch  # noqa: PLC0415
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except ImportError:
        pass

    return video_path
