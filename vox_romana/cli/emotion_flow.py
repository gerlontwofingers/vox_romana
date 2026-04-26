"""Emotion-flow mode runner: cross-lingual emotion analysis or show adaptation.

Extracted from run.py as part of the cli/ refactor so the logic is
importable and unit-testable independently of the master CLI's argparse
layer.
"""

from __future__ import annotations

import argparse
import sys


def run_emotion_flow(args: argparse.Namespace) -> None:
    """Run emotion-flow mode: cross-lingual emotion analysis or show adaptation.

    Without ``--to-show``: loads speech segments from ``--script-input`` (or
    the positional ``prompt``), runs the full EmotionFlowPipeline, and writes
    HTML/JSON/CSV/PNG reports.

    With ``--to-show``: loads a ``report.json`` produced by the analysis pass
    and converts it into a NewsConfig via EmotionFlowShowAdapter.  If
    ``--run-show`` is also set, the show pipeline is executed to produce a
    full show.json.

    Args:
        args: Parsed namespace from run.py's parse_args().
    """
    from vox_romana.core.paths import OUTPUT_DIR as _EF_OUTPUT_DIR

    if args.ef_to_show is not None:
        # ── Show-adapter mode ────────────────────────────────────────
        import json as _json
        from dataclasses import asdict
        from vox_romana.emotion_flow.show_adapter import EmotionFlowShowAdapter

        report_path = args.ef_to_show

        adapter = EmotionFlowShowAdapter(
            n_segments=args.ef_n_segments,
        )

        if not args.quiet:
            print(f"  Loading emotion-flow report: {report_path}")

        config = adapter.adapt_from_file(report_path)

        output_dir = args.emotion_output or (
            _EF_OUTPUT_DIR / "shows" / "emotion_flow"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = output_dir / "show_config.json"
        config_dict = asdict(config)
        config_path.write_text(
            _json.dumps(config_dict, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if not args.quiet:
            print(f"  Show config \u2192 {config_path}")

        if args.ef_run_show:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass

            from vox_romana.core.client import DeepSeekClient
            from vox_romana.shows.news.conductor import NewsConductor
            from vox_romana.shows.news.serializer import NewsSerializer

            if not args.quiet:
                print("  Running show pipeline\u2026")

            client = DeepSeekClient()
            conductor = NewsConductor(config, client=client)  # type: ignore[arg-type]
            result = conductor.run()
            merged = result.to_merged_news_show()

            show_path = output_dir / "show.json"
            NewsSerializer().save(merged, show_path)

            if not args.quiet:
                print(f"  Show \u2192 {show_path}")

    else:
        # ── Analysis mode ────────────────────────────────────────────
        source = args.script_input or args.prompt
        if source is None:
            print(
                "Error: --mode emotion-flow requires --script-input pointing "
                "to a text file, directory, or debate JSON (or use --to-show "
                "PATH for show-adapter mode).",
                file=sys.stderr,
            )
            sys.exit(1)

        from vox_romana.emotion_flow.pipeline import EmotionFlowPipeline

        output_dir = args.emotion_output or (_EF_OUTPUT_DIR / "emotion_flow")

        pipeline = EmotionFlowPipeline(
            output_dir=output_dir,
            seed=args.ef_seed,
            cache_dir=str(args.ef_cache_dir) if args.ef_cache_dir else None,
            n_permutations=args.ef_permutations,
            run_semantic=not args.ef_no_semantic,
            run_translation=not args.ef_no_translate,
        )

        if not args.quiet:
            leader_display = args.ef_leader or source.stem
            print(f"  Source:       {source}")
            print(f"  Leader:       {leader_display}")
            print(f"  Output dir:   {output_dir}")
            print(f"  Seed:         {args.ef_seed}  |  Permutations: {args.ef_permutations}")
            if args.ef_no_translate:
                print("  Translation:  OFF")
            if args.ef_no_semantic:
                print("  Semantic:     OFF")
            print()

        report = pipeline.run(
            source,
            leader=args.ef_leader,
        )

        if not args.quiet and report.report_path:
            print(f"  Report \u2192 {report.report_path}")
