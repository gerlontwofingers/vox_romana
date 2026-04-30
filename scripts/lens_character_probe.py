"""Cast a character across every lens in the rack — Salvatore casting tool.

Generates the *same* monologue prompt through every lens and emits a
side-by-side Markdown report so you can hear how each model would
voice the character.  The headline use-case is what we call the
**Salvatore lens**: a deliberately small, brittle model used to voice
the polyglot lay brother in *The Name of the Rose* who mangles Latin,
Italian and Provençal across every sentence.  A more capable model
*could* play him — but a small one will do it for free, and arguably
more authentically.

USAGE
─────

    python scripts/lens_character_probe.py \
        --character "Salvatore: polyglot lay brother in a 14th-c. abbey, \
mangles Latin/Italian/Provençal in every sentence, fearful, devout" \
        --prompt "Describe what you saw last night in the scriptorium." \
        --out reports/salvatore_probe.md

By default the script writes Markdown to stdout.  ``--lens NAME`` (may
be repeated) restricts the run to specific lenses; otherwise every
registered lens is tried.  Lenses whose backend is unreachable (e.g.
Ollama not running) are reported as ``[unavailable]`` rather than
failing the whole run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

# Allow ``python scripts/lens_character_probe.py`` to find the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vox_romana.core.lens import LLMLens, build_default_rack  # noqa: E402


SYSTEM_TEMPLATE = (
    "You are voicing a single character in a creative work. "
    "Stay strictly in character. Do not narrate or break the fourth wall. "
    "Do not preface with 'Sure' or 'Here is'. Just speak.\n\n"
    "CHARACTER BRIEF:\n{character}\n"
)


def _render_one(lens: LLMLens, character: str, prompt: str, *, max_tokens: int) -> tuple[str, float]:
    """Run a single lens; return (text, latency_seconds).  Never raises."""
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(character=character.strip())},
        {"role": "user", "content": prompt.strip()},
    ]
    t0 = time.perf_counter()
    try:
        client = lens.client  # may raise (e.g. connection refused for ollama)
        text = client.chat(messages, temperature=0.8, max_tokens=max_tokens)
    except Exception as exc:  # broad on purpose — we never abort the probe
        return f"[unavailable: {type(exc).__name__}: {exc}]", time.perf_counter() - t0
    return text.strip(), time.perf_counter() - t0


def _format_markdown(
    character: str,
    prompt: str,
    results: list[tuple[LLMLens, str, float]],
) -> str:
    lines: list[str] = []
    lines.append("# Lens character probe")
    lines.append("")
    lines.append("## Character brief")
    lines.append("")
    lines.append("> " + character.strip().replace("\n", "\n> "))
    lines.append("")
    lines.append("## Prompt")
    lines.append("")
    lines.append("> " + prompt.strip().replace("\n", "\n> "))
    lines.append("")
    for lens, text, dt in results:
        lines.append(f"## Lens — `{lens.name}` ({lens.backend} · {lens.model})")
        lines.append("")
        if lens.personality_hint:
            lines.append(f"*Personality hint:* {lens.personality_hint}")
            lines.append("")
        lines.append(f"*Latency:* {dt:.2f}s · *Cost tier:* {lens.cost_tier}")
        lines.append("")
        lines.append("> " + text.replace("\n", "\n> "))
        lines.append("")
    return "\n".join(lines)


def _select_lenses(rack, only: Iterable[str] | None) -> list[LLMLens]:
    if not only:
        return [rack._lenses[n] for n in rack.list_lenses()]
    missing = [n for n in only if n not in rack._lenses]
    if missing:
        raise SystemExit(f"unknown lens(es): {', '.join(missing)}")
    return [rack._lenses[n] for n in only]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--character", required=True, help="Character brief (1–3 sentences).")
    parser.add_argument("--prompt", required=True, help="What the character should say/do.")
    parser.add_argument("--lens", action="append", help="Restrict to named lens (repeatable).")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--out", type=Path, help="Write Markdown here instead of stdout.")
    args = parser.parse_args(argv)

    rack = build_default_rack()
    lenses = _select_lenses(rack, args.lens)

    results: list[tuple[LLMLens, str, float]] = []
    for lens in lenses:
        print(f"… probing {lens.name}", file=sys.stderr, flush=True)
        text, dt = _render_one(lens, args.character, args.prompt, max_tokens=args.max_tokens)
        results.append((lens, text, dt))

    md = _format_markdown(args.character, args.prompt, results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
