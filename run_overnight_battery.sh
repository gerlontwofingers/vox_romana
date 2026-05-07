#!/usr/bin/env bash
# Overnight battery study — launched 2026-04-30
# 5 show types × 10 topics × 5 min each = 50 pieces
# Log: /tmp/vox_overnight_battery.log
# Results: study_results/overnight_<timestamp>/
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
OUTDIR="study_results/overnight_$(date +%Y%m%d_%H%M%S)"
echo "[overnight] Starting battery → $OUTDIR"
python tools/study_battery.py --duration 5 --out "$OUTDIR"
echo "[overnight] Done. Results in $OUTDIR"
