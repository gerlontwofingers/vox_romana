"""Battery study: generate 10 dry-run pieces per show type for analysis.

Invokes `python run.py --mode generate` as a subprocess for each
(show_type x topic) combination.  No TTS, no MusicGen, no video — just
the script + voice cast + music/SFX cue plan that each conductor
produces today.  Saves everything under
``study_results/<timestamp>/<show>/<slug>/`` for later analysis.

Usage:
    python tools/study_battery.py                # full battery (40 pieces)
    python tools/study_battery.py --shows debate # only one show type
    python tools/study_battery.py --limit 2      # 2 topics per show (smoke)
    python tools/study_battery.py --duration 2   # override target minutes

The runner is deliberately conservative — short durations, fact-check
off, quiet output — so the whole battery completes in reasonable time.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_production_id() -> str:
    """Read production_id from vox_romana.yaml; fall back to 'default'."""
    import yaml  # lazy import — already in requirements.txt
    for cfg_path in [
        REPO_ROOT / "vox_romana.yaml",
        Path.home() / ".vox_romana" / "config.yaml",
    ]:
        if cfg_path.exists():
            try:
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                val = cfg.get("production_id")
                if val:
                    return str(val)
            except Exception:
                pass
    return "default"


def _snapshot_params(study_dir: Path) -> None:
    """Capture all tunable parameter values at run start.

    Imports live constants from the vox_romana package so the snapshot
    reflects the values actually in effect for this battery run, not
    whatever values happen to be current at ingest time later.

    Writes ``study_dir/params_snapshot.json``.
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT))

    try:
        from vox_romana.core.conductor import BaseConductor
        from vox_romana.core.duration import WPM_DEFAULTS
        from vox_romana.tts.renderer import _LEAD_IN_MS
        from vox_romana.shows.news.conductor import MAX_BEAT_WORDS as _NEWS_MAX_BEAT
        from vox_romana.shows.lecture.conductor import MAX_BEAT_WORDS as _LECTURE_MAX_BEAT
        from vox_romana.shows.debates.conductor import DebateConductor
        from vox_romana.cli.runners import (
            _DEBATE_AVG_WORDS_PER_TURN,
            _DEBATE_HOST_OVERHEAD_WORDS,
            _NEWS_AVG_WORDS_PER_SEGMENT,
            _NEWS_HOST_OVERHEAD_WORDS,
        )
    except ImportError as exc:
        print(f"  [snapshot] WARNING: could not import vox_romana constants: {exc}", flush=True)
        return

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_sha = None

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "universal": {
            "overshoot_compensation": BaseConductor.OVERSHOOT_COMPENSATION,
            "wrap_up_threshold": BaseConductor.WRAP_UP_THRESHOLD,
        },
        "by_show": {
            "debate": {
                "wpm_default": WPM_DEFAULTS.get("debate"),
                "lead_in_ms": _LEAD_IN_MS.get("debate"),
                "max_beat_words": None,
                "avg_words_per_turn": _DEBATE_AVG_WORDS_PER_TURN,
                "host_overhead_words": _DEBATE_HOST_OVERHEAD_WORDS,
                "closing_window_extra": DebateConductor.CLOSING_WINDOW_EXTRA,
            },
            "news": {
                "wpm_default": WPM_DEFAULTS.get("news"),
                "lead_in_ms": _LEAD_IN_MS.get("news"),
                "max_beat_words": _NEWS_MAX_BEAT,
                "avg_words_per_segment": _NEWS_AVG_WORDS_PER_SEGMENT,
                "host_overhead_words": _NEWS_HOST_OVERHEAD_WORDS,
            },
            "podcast": {
                "wpm_default": WPM_DEFAULTS.get("podcast"),
                "lead_in_ms": _LEAD_IN_MS.get("podcast"),
                "max_beat_words": None,
            },
            "audiobook": {
                "wpm_default": WPM_DEFAULTS.get("audiobook"),
                "lead_in_ms": _LEAD_IN_MS.get("audiobook"),
                "max_beat_words": None,
            },
            "lecture": {
                "wpm_default": WPM_DEFAULTS.get("lecture"),
                "lead_in_ms": _LEAD_IN_MS.get("lecture"),
                "max_beat_words": _LECTURE_MAX_BEAT,
            },
        },
    }

    (study_dir / "params_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

# ─────────────────────────────────────────────────────────────────────
# Topic battery — 10 per show type, deliberately diverse so we surface
# different conductor behaviours (factual vs fictional, technical vs
# soft, controversial vs benign, narrative vs analytical, etc.)
# ─────────────────────────────────────────────────────────────────────

TOPICS: dict[str, list[str]] = {
    "debate": [
        "Should AI-generated art qualify for copyright protection?",
        "Is universal basic income the right response to automation?",
        "Should genetic engineering of human embryos be legal?",
        "Is nuclear power essential to fighting climate change?",
        "Should social media platforms be liable for user-generated content?",
        "Is space colonization a moral imperative or a distraction?",
        "Should standardized testing be abolished in primary education?",
        "Is geoengineering an acceptable climate intervention?",
        "Should autonomous weapons be banned by international treaty?",
        "Is privacy obsolete in the age of ambient computing?",
        # wave 2
        "Should private space travel be regulated like commercial aviation?",
        "Is the death penalty ever morally justifiable?",
        "Should wealthy nations have open borders for climate refugees?",
        "Is addiction a disease or a moral failing?",
        "Should tech companies be required to provide government backdoor access?",
        "Is it ethical to eat meat in 2026?",
        "Should the voting age be lowered to sixteen?",
        "Is affirmative action still necessary in higher education?",
        "Should performance-enhancing drugs be legal in competitive sports?",
        "Is journalistic objectivity a myth worth preserving?",
    ],
    "news": [
        "Recent advances in fusion reactor design and net-energy milestones",
        "The state of CRISPR therapeutics in 2026",
        "Election security developments after the 2024 cycle",
        "Cybersecurity threats from quantum computing breakthroughs",
        "Climate adaptation strategies in coastal megacities",
        "Antitrust action against major AI labs",
        "Battery technology breakthroughs and EV adoption curves",
        "Mental health crisis among adolescents — research updates",
        "Regulatory response to deepfakes and synthetic media",
        "Microplastic contamination in agricultural soil",
        # wave 2
        "Progress on malaria vaccine deployment in sub-Saharan Africa",
        "Arctic permafrost methane release and climate tipping points",
        "AI-generated scientific papers and the peer review crisis",
        "Post-quantum cryptography standardization rollout",
        "Water scarcity conflicts emerging in Central Asia",
        "Central bank digital currencies and financial privacy concerns",
        "Longevity drugs entering phase-three clinical trials",
        "Satellite megaconstellations and radio astronomy interference",
        "Insurance market collapse in climate-stressed coastal regions",
        "Fentanyl supply chain disruption after major cartel prosecutions",
    ],
    "podcast": [
        "The unexpected history of the metric system",
        "How sourdough conquered the world: a culinary deep-dive",
        "Why we still don't understand consciousness",
        "The economics of attention in the streaming era",
        "What octopus cognition tells us about intelligence",
        "The strange life and legacy of Nikola Tesla",
        "Why cities adopt and abandon transit modes",
        "The mathematics of voting systems",
        "How to think about probability in everyday life",
        "The psychology of conspiracy belief",
        # wave 2
        "The forgotten science behind how anesthesia actually works",
        "Why the Roman Empire took so long to fall",
        "The surprising psychology of color perception",
        "What the history of salt reveals about civilization",
        "How jazz changed the trajectory of twentieth-century music",
        "The physics of why bicycles don't fall over",
        "The strange economics of the used-book market",
        "How languages die and what we lose when they do",
        "The hidden mathematics behind everyday music",
        "Why humans are uniquely bad at assessing long-term risk",
    ],
    "audiobook": [
        "A short story about a librarian who discovers their collection rearranges itself overnight",
        "A speculative fiction piece set in a city where memories are traded as currency",
        "A noir detective vignette in a 1940s seaside town",
        "A children's bedtime story about a shy comet learning to glow",
        "A philosophical dialogue between a gardener and a passing time-traveler",
        "A cozy mystery: the case of the missing prize-winning pumpkin",
        "A first-contact short story told from the alien's perspective",
        "A historical fiction vignette about a scribe in ancient Alexandria",
        "A horror short: a lighthouse keeper discovers the light is watching back",
        "A whimsical fable about a kingdom where everyone speaks in palindromes",
        # wave 2
        "A hard sci-fi short about a crew on the final mission to a dying sun",
        "A magical realism piece about a city where dreams leave physical traces",
        "A tragicomedy about a retired spy trying to live an ordinary life",
        "A lyrical short about the last beekeeper on a depleted earth",
        "A fable about a machine that learned to feel loneliness",
        "A ghost story set in an abandoned library",
        "A comedic vignette about two ghosts arguing over who haunts their house",
        "A dystopian short about the last musician in a world without music",
        "A literary piece about a grandmother teaching her granddaughter to cook",
        "An adventure story for children about a girl who befriends a storm",
    ],
    "lecture": [
        "An introduction to Stoic ethics and how it applies to modern life",
        "A masterclass on the architecture of memory in the human brain",
        "A short course on Bayesian reasoning for decision-making",
        "A lecture on why economic bubbles form and burst",
        "An explainer on how vaccines actually train the immune system",
        "A lecture on the philosophy of language and meaning",
        "A masterclass on negotiation tactics drawn from real cases",
        "A lecture on how black holes warp spacetime",
        "A short course on the history and craft of bread-making",
        "A lecture on cognitive biases and how to spot them in yourself",
        # wave 2
        "A lecture on the history of mathematics from Euclid to Gödel",
        "An introduction to game theory and strategic thinking",
        "A masterclass on the biology of sleep and why it matters",
        "A lecture on how propaganda works and how to resist it",
        "A course unit on the fundamentals of thermodynamics",
        "A lecture on the neuroscience of decision-making under uncertainty",
        "A masterclass on how to read financial statements",
        "A lecture on ancient Greek philosophy: Plato versus Aristotle",
        "A short course on the ethics of artificial intelligence",
        "A lecture on why ecosystems collapse and what restores them",
    ],
}


def _slugify(text: str, max_len: int = 80) -> str:
    """Filesystem-safe slug from free-form text."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "topic"


def _write_brief(path: Path, topic: str, show_type: str) -> None:
    """Write a minimal brief file that the router/bootstrap will accept."""
    path.write_text(topic.strip() + "\n", encoding="utf-8")


def _find_output_json(study_dir: Path, show_type: str, slug: str) -> Path | None:
    """Locate the JSON `run.py --mode generate` wrote into ``study_dir``.

    The CLI auto-names files with the show type and a timestamp; we
    match the show prefix and the brief slug to find the right one.
    """
    if not study_dir.exists():
        return None
    candidates = sorted(
        study_dir.rglob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Most recent JSON in this run's dir is the right one.
    return candidates[0] if candidates else None


def _summarise_script(json_path: Path) -> dict:
    """Pull lightweight stats out of a generated show JSON."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": f"could not parse JSON: {exc}"}

    summary: dict = {
        "title": data.get("title") or data.get("show_title") or "",
        "show_type": data.get("show_type") or data.get("type") or "",
    }

    # Common shapes across show types: events, utterances, segments, beats.
    events: list = (
        data.get("events")
        or data.get("utterances")
        or data.get("segments")
        or data.get("beats")
        or []
    )
    summary["event_count"] = len(events) if isinstance(events, list) else 0
    if isinstance(events, list) and events:
        word_count = 0
        speakers: set[str] = set()
        nonverbal_total = 0
        nonverbal_uniq: set[str] = set()
        for ev in events:
            if not isinstance(ev, dict):
                continue
            text = ev.get("text") or ""
            if isinstance(text, str):
                word_count += len(text.split())
            spk = (
                ev.get("speaker_name")
                or ev.get("speaker")
                or ev.get("voice")
                or ev.get("character")
                or ev.get("narrator_name")
            )
            if spk:
                speakers.add(str(spk))
            nv = ev.get("nonverbal") or []
            if isinstance(nv, list):
                nonverbal_total += len(nv)
                for tag in nv:
                    if isinstance(tag, str):
                        nonverbal_uniq.add(tag)
        summary["word_count"] = word_count
        summary["speakers"] = sorted(speakers)
        summary["nonverbal_cue_count"] = nonverbal_total
        summary["nonverbal_unique"] = sorted(nonverbal_uniq)

    # Voice cast lives in `voice_casting` (dict: name -> gender/hint string)
    # for all current show types.  Fall back to `participants`/`cast` when
    # they're present as structured dicts (older shapes).
    cast: list[dict] = []
    voice_casting = data.get("voice_casting")
    if isinstance(voice_casting, dict):
        for name, info in voice_casting.items():
            if isinstance(info, dict):
                cast.append({
                    "name": name,
                    "voice_id": info.get("voice_id") or info.get("voice", ""),
                    "gender": info.get("gender", ""),
                })
            else:
                cast.append({"name": name, "voice_id": "", "gender": str(info)})
    else:
        participants = data.get("participants") or data.get("cast") or []
        if isinstance(participants, list):
            for p in participants:
                if isinstance(p, dict):
                    cast.append({
                        "name": p.get("name", ""),
                        "voice_id": p.get("voice_id") or p.get("voice", ""),
                        "role": p.get("role", ""),
                    })
                elif isinstance(p, str):
                    cast.append({"name": p, "voice_id": "", "role": ""})
    summary["cast"] = cast
    summary["cast_size"] = len(cast)

    # Voice hints (per-character delivery suggestions) — useful for
    # later audit of how varied the bootstrap's directing actually is.
    vh = data.get("voice_hints")
    if isinstance(vh, dict):
        summary["voice_hint_count"] = len(vh)

    return summary


def _run_one(
    *,
    show_type: str,
    topic: str,
    study_dir: Path,
    duration: float,
    quiet: bool,
) -> dict:
    """Run one piece. Returns telemetry dict."""
    slug = _slugify(topic)
    piece_dir = study_dir / show_type / slug
    piece_dir.mkdir(parents=True, exist_ok=True)
    brief_path = piece_dir / "brief.txt"
    _write_brief(brief_path, topic, show_type)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "run.py"),
        str(brief_path),
        "--mode", "generate",
        "--type", show_type,
        "--output", str(piece_dir),
        "--duration", str(duration),
        "--fact-check", "off",
        "--no-research",
        "--quiet",
    ]

    started = time.monotonic()
    started_iso = datetime.now(timezone.utc).isoformat()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=900,  # 15 min hard cap per piece
            check=False,
        )
        rc = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        rc = -1
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "\n[battery] subprocess timed out after 900s"
        timed_out = True
    elapsed = time.monotonic() - started

    # Save raw stdout/stderr for forensics.
    (piece_dir / "stdout.log").write_text(stdout or "", encoding="utf-8")
    (piece_dir / "stderr.log").write_text(stderr or "", encoding="utf-8")

    json_path = _find_output_json(piece_dir, show_type, slug)
    summary: dict = {}
    if json_path and json_path.exists():
        summary = _summarise_script(json_path)

    telemetry = {
        "show_type": show_type,
        "topic": topic,
        "slug": slug,
        "started_at": started_iso,
        "elapsed_secs": round(elapsed, 2),
        "exit_code": rc,
        "timed_out": timed_out,
        "ok": (rc == 0) and (json_path is not None),
        "json_path": str(json_path.relative_to(study_dir)) if json_path else None,
        "stderr_tail": (stderr or "").splitlines()[-10:] if stderr else [],
        "summary": summary,
    }
    (piece_dir / "telemetry.json").write_text(
        json.dumps(telemetry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    status = "ok" if telemetry["ok"] else f"FAIL(rc={rc})"
    if not quiet:
        print(
            f"  [{show_type:9s}] {status:8s} {elapsed:6.1f}s  {topic[:60]}",
            flush=True,
        )
    return telemetry


def _build_index(study_dir: Path, results: list[dict], duration: float | None = None) -> None:
    """Aggregate index + per-show stats."""
    by_show: dict[str, list[dict]] = {}
    for r in results:
        by_show.setdefault(r["show_type"], []).append(r)

    aggregate = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_duration_mins": duration,
        "total_pieces": len(results),
        "total_ok": sum(1 for r in results if r["ok"]),
        "total_failed": sum(1 for r in results if not r["ok"]),
        "total_elapsed_secs": round(sum(r["elapsed_secs"] for r in results), 2),
        "by_show": {},
    }
    for show, items in by_show.items():
        ok = [r for r in items if r["ok"]]
        words = [r["summary"].get("word_count", 0) for r in ok if r.get("summary")]
        events = [r["summary"].get("event_count", 0) for r in ok if r.get("summary")]
        aggregate["by_show"][show] = {
            "n": len(items),
            "ok": len(ok),
            "failed": len(items) - len(ok),
            "elapsed_secs": round(sum(r["elapsed_secs"] for r in items), 2),
            "avg_word_count": round(sum(words) / len(words), 1) if words else 0,
            "avg_event_count": round(sum(events) / len(events), 1) if events else 0,
        }

    (study_dir / "index.json").write_text(
        json.dumps({"aggregate": aggregate, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("═" * 64)
    print(f"BATTERY COMPLETE — {aggregate['total_ok']}/{aggregate['total_pieces']} ok")
    print(f"Total elapsed: {aggregate['total_elapsed_secs']:.1f}s")
    print("─" * 64)
    print(f"{'show':12s} {'ok/n':>8s} {'elapsed':>10s} {'avg_words':>12s} {'avg_events':>12s}")
    for show, stats in aggregate["by_show"].items():
        print(
            f"{show:12s} {stats['ok']}/{stats['n']:<6d} "
            f"{stats['elapsed_secs']:>9.1f}s "
            f"{stats['avg_word_count']:>12.1f} "
            f"{stats['avg_event_count']:>12.1f}"
        )
    print("═" * 64)
    print(f"Saved → {study_dir}/index.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Battery study: generate dry-run pieces across show types.",
    )
    parser.add_argument(
        "--shows",
        nargs="+",
        choices=list(TOPICS.keys()),
        default=list(TOPICS.keys()),
        help="Which show types to run (default: all five).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap topics per show (default: all 10). Useful for smoke runs.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Target duration in minutes per piece (default: 3.0).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: study_results/<timestamp>/).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-piece progress lines.",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of pieces to generate in parallel (default: 1 — sequential). "
            "Each worker is an independent subprocess, so this controls how many "
            "concurrent LLM calls are in flight. Recommended on M1 Max: 2–3."
        ),
    )
    parser.add_argument(
        "--production-id",
        default=None,
        metavar="ID",
        help=(
            "Production identifier written into longitudinal.db when "
            "--auto-ingest is set (default: production_id from vox_romana.yaml)."
        ),
    )
    parser.add_argument(
        "--auto-ingest",
        action="store_true",
        default=False,
        help=(
            "After the battery completes, automatically ingest the results "
            "into study_results/longitudinal.db via ingest_run()."
        ),
    )
    args = parser.parse_args(argv)

    if args.workers < 1:
        print("Error: --workers must be >= 1.", file=sys.stderr)
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = args.out or (REPO_ROOT / "study_results" / ts)
    study_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_params(study_dir)

    pieces: list[tuple[str, str]] = []
    for show in args.shows:
        topics = TOPICS[show]
        if args.limit:
            topics = topics[: args.limit]
        for topic in topics:
            pieces.append((show, topic))

    print(f"Battery study → {study_dir}")
    print(f"  shows: {', '.join(args.shows)}")
    print(f"  pieces: {len(pieces)}  (target {args.duration:.1f} min each)")
    if args.workers > 1:
        print(f"  workers: {args.workers}  (parallel)")
    print()

    if args.workers == 1:
        # Sequential path — backward-compatible, preserves existing behaviour.
        results: list[dict] = []
        for i, (show, topic) in enumerate(pieces, 1):
            if not args.quiet:
                print(f"[{i:>3d}/{len(pieces)}]", end=" ", flush=True)
            results.append(
                _run_one(
                    show_type=show,
                    topic=topic,
                    study_dir=study_dir,
                    duration=args.duration,
                    quiet=args.quiet,
                )
            )
    else:
        # Parallel path — submit all pieces, collect in original study order.
        def _run_and_tag(idx: int, show: str, topic: str) -> tuple[int, dict]:
            return idx, _run_one(
                show_type=show,
                topic=topic,
                study_dir=study_dir,
                duration=args.duration,
                quiet=True,  # progress handled below on completion
            )

        slots: dict[int, dict] = {}
        completed = 0
        n = len(pieces)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_run_and_tag, i, show, topic): i
                for i, (show, topic) in enumerate(pieces)
            }
            for future in concurrent.futures.as_completed(futures):
                idx, result = future.result()
                slots[idx] = result
                completed += 1
                if not args.quiet:
                    status = "ok" if result["ok"] else f"FAIL(rc={result['exit_code']})"
                    print(
                        f"  [{completed:>3d}/{n}] [{result['show_type']:9s}] "
                        f"{status:8s} {result['elapsed_secs']:6.1f}s  "
                        f"{result['topic'][:60]}",
                        flush=True,
                    )
        results = [slots[i] for i in range(len(pieces))]

    _build_index(study_dir, results, duration=args.duration)

    if args.auto_ingest:
        production_id = args.production_id or _read_production_id()
        print(f"\nAuto-ingesting into longitudinal.db (production_id={production_id!r})...")
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from vox_romana.analysis.longitudinal import ingest_run
            inserted = ingest_run(
                db_path=REPO_ROOT / "study_results" / "longitudinal.db",
                study_dir=study_dir,
                production_id=production_id,
            )
            total = sum(inserted.values())
            print(f"  Ingested {total} pieces: {dict(sorted(inserted.items()))}")
        except Exception as exc:
            print(f"  [auto-ingest] WARNING: {exc}", file=sys.stderr)

    # Failure summary — surface which pieces failed without digging into subdirs.
    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\n  {len(failed)} piece(s) failed:")
        for r in failed:
            tail = r.get("stderr_tail") or []
            tail_str = tail[-1] if tail else "(no stderr)"
            print(f"    [{r['show_type']:9s}] {r['topic'][:60]}")
            print(f"               rc={r['exit_code']} | {tail_str}")
        print()

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
