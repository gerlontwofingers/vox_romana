"""Lens parity report — feed identical structured prompts to every lens.

Used to spot-check that the local lenses can stand in for the cloud
lens on the *structured* tasks vox_romana relies on (debate bootstrap,
news outline, conductor turn).  Reports JSON-validity per lens so we
know which lens to assign to which role in the rack.

USAGE
─────

    python scripts/lens_parity.py                 # built-in prompt set
    python scripts/lens_parity.py --prompts my.json
    python scripts/lens_parity.py --out reports/parity.md

The prompts file is a JSON list of objects::

    [
      {"name": "...", "system": "...", "user": "...", "json": true},
      ...
    ]

When ``json`` is true the prompt is sent with
``response_format={"type":"json_object"}`` and the response is checked
for ``json.loads`` validity.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Allow ``python scripts/lens_parity.py`` to find the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vox_romana.core.lens import LLMLens, build_default_rack  # noqa: E402


# Built-in smoke-test set — small, structured, representative of real
# vox_romana call-sites.
_DEFAULT_PROMPTS: list[dict[str, Any]] = [
    {
        "name": "debate_bootstrap",
        "system": "You design philosophical debates. Reply ONLY with valid JSON.",
        "user": (
            "Design a 2-speaker debate on whether silence is a form of speech. "
            'Reply as JSON: {"title": str, "speakers": [{"name": str, "stance": str}]}.'
        ),
        "json": True,
    },
    {
        "name": "news_outline",
        "system": "You are a news editor. Reply ONLY with valid JSON.",
        "user": (
            "Outline a 3-segment podcast about a new exoplanet discovery. "
            'JSON: {"segments": [{"heading": str, "beats": [str]}]}.'
        ),
        "json": True,
    },
    {
        "name": "conductor_turn",
        "system": "You write one short turn of dialogue.",
        "user": (
            "Write Speaker B's reply (≤2 sentences) to: "
            '"A: But surely art needs constraint to flourish?"'
        ),
        "json": False,
    },
]


def _run_prompt(lens: LLMLens, prompt: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]
    kwargs: dict[str, Any] = {"temperature": 0.3, "max_tokens": max_tokens}
    if prompt.get("json"):
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.perf_counter()
    try:
        text = lens.client.chat(messages, **kwargs)
        dt = time.perf_counter() - t0
    except Exception as exc:
        return {
            "ok": False,
            "latency": time.perf_counter() - t0,
            "error": f"{type(exc).__name__}: {exc}",
            "text": "",
            "json_valid": None,
        }

    json_valid: bool | None = None
    if prompt.get("json"):
        try:
            json.loads(text)
            json_valid = True
        except (ValueError, TypeError):
            json_valid = False

    return {
        "ok": True,
        "latency": dt,
        "error": None,
        "text": text.strip(),
        "json_valid": json_valid,
    }


def _format_report(prompts: list[dict[str, Any]], rows: dict[str, dict[str, dict[str, Any]]]) -> str:
    out: list[str] = ["# Lens parity report", ""]
    out.append("## Summary")
    out.append("")
    out.append("| Lens | " + " | ".join(p["name"] for p in prompts) + " |")
    out.append("|" + "---|" * (len(prompts) + 1))
    for lens_name, results in rows.items():
        cells = []
        for p in prompts:
            r = results[p["name"]]
            if not r["ok"]:
                cells.append("✗ err")
            elif r["json_valid"] is False:
                cells.append(f"~ {r['latency']:.1f}s (bad JSON)")
            elif r["json_valid"] is True:
                cells.append(f"✓ {r['latency']:.1f}s")
            else:
                cells.append(f"· {r['latency']:.1f}s")
        out.append(f"| `{lens_name}` | " + " | ".join(cells) + " |")
    out.append("")
    out.append("Legend: ✓ valid JSON · ~ invalid JSON · · text-mode (no check) · ✗ exception")
    out.append("")
    out.append("## Detail")
    out.append("")
    for lens_name, results in rows.items():
        out.append(f"### `{lens_name}`")
        out.append("")
        for p in prompts:
            r = results[p["name"]]
            out.append(f"**{p['name']}** — {r['latency']:.2f}s")
            if r["error"]:
                out.append("")
                out.append(f"`error:` {r['error']}")
            else:
                out.append("")
                out.append("```")
                out.append(r["text"][:1200])
                out.append("```")
            out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, help="JSON list of prompts (default: built-in set).")
    parser.add_argument("--lens", action="append", help="Restrict to named lens (repeatable).")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--out", type=Path, help="Write Markdown report here instead of stdout.")
    args = parser.parse_args(argv)

    prompts = _DEFAULT_PROMPTS
    if args.prompts:
        prompts = json.loads(args.prompts.read_text(encoding="utf-8"))

    rack = build_default_rack()
    names = list(rack._lenses.keys())
    if args.lens:
        unknown = [n for n in args.lens if n not in names]
        if unknown:
            raise SystemExit(f"unknown lens(es): {', '.join(unknown)}")
        names = list(args.lens)

    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for name in names:
        lens = rack._lenses[name]
        rows[name] = {}
        for p in prompts:
            print(f"… {name} / {p['name']}", file=sys.stderr, flush=True)
            rows[name][p["name"]] = _run_prompt(lens, p, args.max_tokens)

    md = _format_report(prompts, rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
