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

        # ── Core pip packages ─────────────────────────────────────────
        check_import("openai",           required=True),
        check_import("dotenv",           pkg_name="python-dotenv", required=True),
        check_import("yaml",             pkg_name="pyyaml",        required=True),
        check_import("pydub",            required=True),
        check_import("numpy",            required=True),
        check_import("chromadb",         required=True),

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
    ]


# ── --fix: install missing pip packages ───────────────────────────────────────

def do_fix() -> None:
    req = Path(__file__).parent / "requirements.txt"
    if not req.exists():
        print("requirements.txt not found — cannot auto-fix.", file=sys.stderr)
        sys.exit(1)
    print(f"Running: pip install -r {req}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


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
