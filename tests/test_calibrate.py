from __future__ import annotations

import sqlite3
from pathlib import Path

import yaml

from tools.calibrate import build_recommendations, run_calibration
from vox_romana.analysis.longitudinal import init_db


def _insert_run(
    db_path: Path,
    *,
    run_id: str,
    show_type: str,
    production_id: str = "default",
    target_duration_mins: float = 3.0,
    target_words: int = 450,
    overshoot_compensation: float = 0.92,
    max_beat_words: int | None = None,
    avg_words_per_turn: int | None = None,
    avg_words_per_segment: int | None = None,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO runs (
                run_id, show_type, production_id, target_duration_mins, target_words,
                overshoot_compensation, max_beat_words,
                avg_words_per_turn, avg_words_per_segment,
                n_pieces, n_ok
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                show_type,
                production_id,
                target_duration_mins,
                target_words,
                overshoot_compensation,
                max_beat_words,
                avg_words_per_turn,
                avg_words_per_segment,
                1,
                1,
            ),
        )
        conn.commit()


def _insert_piece(
    db_path: Path,
    *,
    run_id: str,
    show_type: str,
    production_id: str = "default",
    target_duration_mins: float = 3.0,
    target_words: int = 450,
    word_ratio: float = 1.0,
    architect_section_count: int = 3,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO pieces (
                run_id, show_type, production_id,
                target_duration_mins, target_words,
                architect_section_count,
                word_ratio, exit_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                run_id,
                show_type,
                production_id,
                target_duration_mins,
                target_words,
                architect_section_count,
                word_ratio,
            ),
        )
        conn.commit()


def test_ratio_update_reduces_overshoot_for_small_sample(tmp_path: Path) -> None:
    db_path = tmp_path / "longitudinal.db"
    init_db(db_path)
    _insert_run(db_path, run_id="run1", show_type="podcast", overshoot_compensation=0.92)
    _insert_piece(db_path, run_id="run1", show_type="podcast", word_ratio=2.0)

    recs = build_recommendations(db_path, "default")

    assert len(recs) == 1
    rec = recs[0]
    assert rec.strategy == "ratio"
    assert rec.recommended["overshoot_compensation"] == 0.46


def test_binding_detection_blocks_debate_multiplier(tmp_path: Path) -> None:
    db_path = tmp_path / "longitudinal.db"
    init_db(db_path)
    _insert_run(
        db_path,
        run_id="run1",
        show_type="debate",
        target_duration_mins=1.0,
        target_words=150,
        overshoot_compensation=0.92,
        avg_words_per_turn=40,
    )
    _insert_piece(
        db_path,
        run_id="run1",
        show_type="debate",
        target_duration_mins=1.0,
        target_words=150,
        word_ratio=2.887,
        architect_section_count=2,
    )

    rec = build_recommendations(db_path, "default")[0]

    assert rec.strategy == "binding"
    assert rec.recommended["overshoot_compensation"] == 0.92
    assert any("format constraint" in note.lower() for note in rec.notes)


def test_ols_recommendation_uses_parameter_variation(tmp_path: Path) -> None:
    db_path = tmp_path / "longitudinal.db"
    init_db(db_path)

    for idx in range(10):
        run_id = f"run{idx:02d}"
        overshoot = 0.60 + (idx * 0.05)
        ratio = 2.0 - overshoot
        _insert_run(
            db_path,
            run_id=run_id,
            show_type="podcast",
            overshoot_compensation=overshoot,
        )
        _insert_piece(
            db_path,
            run_id=run_id,
            show_type="podcast",
            word_ratio=ratio,
            architect_section_count=3,
        )

    rec = build_recommendations(db_path, "default")[0]

    assert rec.strategy == "ols"
    assert rec.r_squared is not None
    assert abs(rec.recommended["overshoot_compensation"] - 1.0) < 0.01


def test_global_baseline_used_when_local_production_has_no_data(tmp_path: Path) -> None:
    db_path = tmp_path / "longitudinal.db"
    init_db(db_path)
    _insert_run(
        db_path,
        run_id="run1",
        show_type="news",
        production_id="prod_a",
        overshoot_compensation=0.91,
        max_beat_words=80,
        avg_words_per_segment=180,
    )
    _insert_piece(
        db_path,
        run_id="run1",
        show_type="news",
        production_id="prod_a",
        word_ratio=1.0,
    )

    rec = build_recommendations(db_path, "new_prod")[0]

    assert rec.strategy == "global_baseline"
    assert rec.source == "global_baseline"
    assert rec.recommended["max_beat_words"] == 80
    assert rec.recommended["avg_words_per_segment"] == 180


def test_apply_persists_state_and_yaml(tmp_path: Path) -> None:
    db_path = tmp_path / "longitudinal.db"
    output_path = tmp_path / "calibration.yaml"
    init_db(db_path)
    _insert_run(db_path, run_id="run1", show_type="podcast", overshoot_compensation=0.92)
    _insert_piece(db_path, run_id="run1", show_type="podcast", word_ratio=2.0)

    exit_code = run_calibration(
        db_path=db_path,
        production_id="default",
        apply=True,
        output_path=output_path,
    )

    assert exit_code == 0
    assert output_path.exists()
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["calibration"]["podcast"]["overshoot_compensation"] == 0.46

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT overshoot_compensation, n_runs_used
            FROM calibration_state
            WHERE production_id = 'default' AND show_type = 'podcast'
            """
        ).fetchone()

    assert row == (0.46, 1)