#!/usr/bin/env python3
"""preflight.py — vox_romana environment health check.

Usage:
    python preflight.py           # human-readable report
    python preflight.py --fix     # auto-install missing pip packages
    python preflight.py --quiet   # JSON output for CI

Exit codes:
    0  all required checks pass
    1  one or more required checks failed
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class Result:
    name: str
    ok: bool
    note: str = ""
    required: bool = True
    info_only: bool = False   # show regardless, no pass/fail semantics


# ── Individual checks ─────────────────────────────────────────────────────────

def check_python_version() -> Result:
    vi = sys.version_info
    ok = (vi.major, vi.minor) >= (3, 12)
    note = f"{vi.major}.{vi.minor}.{vi.micro}"
    if not ok:
        note += " — need ≥3.12"
    return Result("Python ≥3.12", ok, note)


def check_tool(cmd: str, *, required: bool = True, args: list[str] | None = None) -> Result:
    path = shutil.which(cmd)
    if path is None:
        return Result(cmd, False, "not found on PATH", required=required)
    if args:
        try:
            out = subprocess.check_output([cmd] + args, stderr=subprocess.STDOUT, text=True)
            note = out.strip().splitlines()[0][:80]
        except Exception:
            note = path
    else:
        note = path
    return Result(cmd, True, note, required=required)


def check_import(module: str, *, pkg_name: str | None = None, required: bool = True) -> Result:
    label = pkg_name or module
    try:
        importlib.import_module(module)
        return Result(label, True, "importable", required=required)
    except ImportError as exc:
        return Result(label, False, str(exc)[:120], required=required)


def check_dotenv_and_api_key() -> Result:
    """Require .env to exist and contain a non-empty DEEPSEEK_API_KEY."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return Result(".env + DEEPSEEK_API_KEY", False, ".env not found — copy .env.example")

    # Parse without python-dotenv so we don't depend on it being installed yet.
    key_found = False
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY") and "=" in line:
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    key_found = True
                    break
    except OSError as exc:
        return Result(".env + DEEPSEEK_API_KEY", False, str(exc))

    if not key_found:
        return Result(".env + DEEPSEEK_API_KEY", False, "DEEPSEEK_API_KEY missing or empty in .env")
    return Result(".env + DEEPSEEK_API_KEY", True, ".env present, key set")


def check_vox_romana_yaml() -> Result:
    """Informational: note whether vox_romana.yaml is present."""
    yaml_path = Path(__file__).parent / "vox_romana.yaml"
    if yaml_path.exists():
        note = str(yaml_path)
    else:
        note = "not found — using defaults (copy vox_romana.example.yaml to customise)"
    return Result("vox_romana.yaml", yaml_path.exists(), note, required=False)


def check_piper_voices() -> Result:
    """At least one .onnx voice file must exist under PIPER_VOICES_DIR."""
    try:
        from vox_romana.core.paths import PIPER_VOICES_DIR
    except Exception as exc:
        return Result("piper voices (.onnx)", False, f"could not import paths: {exc}")

    if not PIPER_VOICES_DIR.exists():
        return Result(
            "piper voices (.onnx)", False,
            f"{PIPER_VOICES_DIR} not found — clone rhasspy/piper-voices via git-lfs",
        )
    voices = list(PIPER_VOICES_DIR.rglob("*.onnx"))
    if not voices:
        return Result(
            "piper voices (.onnx)", False,
            f"{PIPER_VOICES_DIR} exists but contains no .onnx files",
        )
    return Result("piper voices (.onnx)", True, f"{len(voices)} voice(s) in {PIPER_VOICES_DIR}")


def check_chroma_onnx_model() -> Result:
    """ChromaDB's default embedding model (all-MiniLM-L6-v2 ONNX) must be cached.

    The model is downloaded automatically on first use, but that download
    happens inside tests and can take >30s — causing apparent hangs.
    Run ``python preflight.py --fix`` once to pre-cache it.
    """
    try:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
            ONNXMiniLM_L6_V2,
        )
        from pathlib import Path as _Path
        ef = ONNXMiniLM_L6_V2()
        model_dir = _Path(ef.DOWNLOAD_PATH) / ef.EXTRACTED_FOLDER_NAME
        model_file = model_dir / "model.onnx"
        if model_file.exists():
            return Result(
                "chroma ONNX model",
                True,
                str(model_file),
                required=False,
            )
        return Result(
            "chroma ONNX model",
            False,
            f"not cached at {model_dir} — run: python preflight.py --fix",
            required=False,
        )
    except Exception as exc:
        return Result(
            "chroma ONNX model",
            False,
            f"could not check: {exc}",
            required=False,
        )


def fix_chroma_onnx_model() -> None:
    """Download and warm-up the ChromaDB ONNX embedding model."""
    try:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
            ONNXMiniLM_L6_V2,
        )
        print("Warming up ChromaDB ONNX embedding model (first-time download)…")
        ef = ONNXMiniLM_L6_V2()
        ef(["warm-up"])  # triggers download + onnxruntime session init
        print("  ✓ chroma ONNX model cached.")
    except Exception as exc:
        print(f"  ✗ Failed to cache chroma ONNX model: {exc}", file=sys.stderr)


def check_hf_cache() -> Result:
    """Informational: show where the HuggingFace model cache lives."""
    try:
        from vox_romana.core.paths import HF_CACHE_DIR
        note = str(HF_CACHE_DIR)
        exists = HF_CACHE_DIR.exists()
        note += " (present)" if exists else " (not yet created)"
    except Exception:
        note = os.environ.get("HF_HOME", "~/.cache/huggingface") + " (default)"
    return Result("HF cache dir", True, note, info_only=True)


# ── Video / animation checks (--mode video) ───────────────────────────────────

def check_imagemagick() -> Result:
    """ImageMagick binary — needed by moviepy TextClip for text overlays.

    Not required: the video renders without it (text overlays are silently
    skipped).  Install with: brew install imagemagick
    """
    # ImageMagick 7 ships a unified `magick` binary; legacy `convert` is a
    # symlink.  Check `magick` first; fall back to `convert` so older
    # installations are also recognised.
    for binary in ("magick", "convert"):
        path = shutil.which(binary)
        if path:
            try:
                out = subprocess.check_output(
                    [binary, "--version"], stderr=subprocess.STDOUT, text=True
                )
                version_line = out.strip().splitlines()[0][:80]
            except Exception:
                version_line = path
            return Result(
                "imagemagick",
                True,
                f"{version_line} ({binary})",
                required=False,
            )
    return Result(
        "imagemagick",
        False,
        "not found — text overlays will be skipped (brew install imagemagick)",
        required=False,
    )


def check_video_python_deps() -> list[Result]:
    """Return Results for all Python packages required by --mode video."""
    return [
        check_import(
            "diffusers",
            required=False,
        ),
        check_import(
            "moviepy",
            pkg_name="moviepy",
            required=False,
        ),
        check_import(
            "PIL",
            pkg_name="Pillow",
            required=False,
        ),
    ]


def check_ollama_models() -> list[Result]:
    """Check whether the Ollama models used by the University module are pulled.

    Returns an empty list when Ollama is not installed so that the overall
    preflight still passes for users who don't use the University feature.

    Required for University:
        deepseek-r1:14b — synthesis / finalize / colloquium (synthesis_model)
        deepseek-r1:7b  — per-article note extraction (reading_model)

    Optional (lens rack):
        qwen2.5:7b  — polyglot / translator lens
        llama3.2:3b — Salvatore / dim lens
    """
    if shutil.which("ollama") is None:
        return []

    try:
        out = subprocess.check_output(
            ["ollama", "list"], stderr=subprocess.STDOUT, text=True, timeout=10
        )
    except Exception as exc:
        return [Result(
            "ollama models",
            False,
            f"could not run 'ollama list': {exc}",
            required=False,
        )]

    # Parse pulled model names from the list output (first column, after the header).
    pulled: set[str] = set()
    for line in out.splitlines()[1:]:
        cols = line.split()
        if cols:
            pulled.add(cols[0].lower())

    def _model(model: str, label: str, *, required: bool) -> Result:
        key = model.lower()
        present = key in pulled
        if present:
            return Result(label, True, f"{model} — pulled", required=required)
        return Result(
            label, False,
            f"{model} not found — run: ollama pull {model}",
            required=required,
        )

    return [
        _model(
            "deepseek-r1:14b",
            "ollama: deepseek-r1:14b (university synthesis)",
            required=False,
        ),
        _model(
            "deepseek-r1:7b",
            "ollama: deepseek-r1:7b  (university reading)",
            required=False,
        ),
        _model(
            "qwen2.5:7b",
            "ollama: qwen2.5:7b (polyglot lens)",
            required=False,
        ),
        _model(
            "llama3.2:3b",
            "ollama: llama3.2:3b (Salvatore lens)",
            required=False,
        ),
    ]


# ── Run all checks ────────────────────────────────────────────────────────────

def _run_all() -> list[Result]:
    return [
        # ── Python ────────────────────────────────────────────────────
        check_python_version(),

        # ── System tools ──────────────────────────────────────────────
        check_tool("ffmpeg",    required=True,  args=["-version"]),
        check_tool("espeak-ng", required=True,  args=["--version"]),
        check_tool("git-lfs",   required=True,  args=["version"]),
        check_tool("ollama",    required=False, args=["--version"]),

        # ── Ollama model availability (University + Lens Rack) ────────
        *check_ollama_models(),

        # ── Core pip packages ─────────────────────────────────────────
        check_import("openai",           required=True),
        check_import("dotenv",           pkg_name="python-dotenv", required=True),
        check_import("yaml",             pkg_name="pyyaml",        required=True),
        check_import("pydub",            required=True),
        check_import("numpy",            required=True),
        check_import("chromadb",         required=True),
        check_chroma_onnx_model(),

        # ── Optional but strongly recommended ─────────────────────────
        check_import("mlx",              required=False),
        check_import("transformers",     required=False),
        check_import("hf_transfer",      required=False),
        check_import("torch",            required=False),
        check_import("piper",            pkg_name="piper-tts",     required=False),

        # ── Config / secrets ──────────────────────────────────────────
        check_dotenv_and_api_key(),
        check_vox_romana_yaml(),

        # ── Data / models ─────────────────────────────────────────────
        check_piper_voices(),
        check_hf_cache(),

        # ── Animation / video (--mode video) ─────────────────────────
        *check_video_python_deps(),
        check_imagemagick(),
    ]


# ── --fix: install missing pip packages ───────────────────────────────────────

def do_fix() -> None:
    req = Path(__file__).parent / "requirements.txt"
    if not req.exists():
        print("requirements.txt not found — cannot auto-fix.", file=sys.stderr)
        sys.exit(1)
    print(f"Running: pip install -r {req}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])
    fix_chroma_onnx_model()


# ── Rendering ─────────────────────────────────────────────────────────────────

_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


def _icon(r: Result) -> str:
    if r.info_only:
        return f"{_CYAN}ℹ{_RESET}"
    if r.ok:
        return f"{_GREEN}✓{_RESET}"
    if r.required:
        return f"{_RED}✗{_RESET}"
    return f"{_YELLOW}⚠{_RESET}"


def print_report(results: list[Result]) -> int:
    """Print human-readable report. Returns exit code."""
    print()
    print(f"{_BOLD}vox_romana preflight check{_RESET}")
    print("─" * 50)

    required_failed: list[Result] = []
    optional_warned: list[Result] = []

    for r in results:
        label = f"{r.name:<32}"
        print(f"  {_icon(r)}  {label}  {r.note}")
        if not r.ok and not r.info_only:
            if r.required:
                required_failed.append(r)
            else:
                optional_warned.append(r)

    print()
    if required_failed:
        print(f"{_RED}{_BOLD}FAILED{_RESET} — {len(required_failed)} required check(s) did not pass:")
        for r in required_failed:
            print(f"    • {r.name}: {r.note}")
        print()
        return 1

    if optional_warned:
        print(f"{_YELLOW}READY{_RESET} — optional items missing: " +
              ", ".join(r.name for r in optional_warned))
    else:
        print(f"{_GREEN}{_BOLD}ALL CHECKS PASSED{_RESET}")
    print()
    return 0


def print_json(results: list[Result]) -> int:
    """Print JSON report. Returns exit code."""
    payload = {
        "checks": [
            {
                "name": r.name,
                "ok": r.ok,
                "note": r.note,
                "required": r.required,
                "info_only": r.info_only,
            }
            for r in results
        ],
        "passed": all(r.ok for r in results if r.required and not r.info_only),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that the vox_romana environment is ready to run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit 0 = all required checks pass\n"
            "Exit 1 = one or more required checks failed\n\n"
            "Use --fix to auto-install pip packages.\n"
            "Use --quiet for JSON output suitable for CI scripts."
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="run 'pip install -r requirements.txt' to resolve missing packages",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="output JSON instead of human-readable text",
    )
    args = parser.parse_args()

    if args.fix:
        do_fix()
        print()

    results = _run_all()

    if args.quiet:
        exit_code = print_json(results)
    else:
        exit_code = print_report(results)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
