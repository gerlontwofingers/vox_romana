# Long-Form Battery Study Plan — 30-Minute Shows
# Created: 2026-05-01 | Status: READY TO EXECUTE

## Goal

Run a second overnight battery at 30-minute show duration to compare against the
5-minute (overnight_20260430_211235) and 3-minute (20260430_195715) datasets.
Data collection only — analysis deferred to a future session.

Key questions this study answers:
- Do name attractors (Dr. Amara Osei, Kai Nakamura) compound further at long duration?
- Does the overshoot ratio get worse / better / stable as duration scales?
- Does lecture maintain coherence over a 30-min single-speaker format?
- Does pace monotony worsen at longer content (more time = more repetition)?
- Does the conductor generate more events or just longer ones?
- Are voice hints / nonverbal cues more or less present in long-form?

## Launch command

```bash
cd /Users/bagheera/repos/vox_romana
source .venv/bin/activate
OUTDIR="study_results/longform_$(date +%Y%m%d_%H%M%S)"
nohup python tools/study_battery.py \
  --duration 30 \
  --limit 10 \
  --out "$OUTDIR" \
  > /tmp/vox_longform_battery.log 2>&1 &
echo "PID=$! OUTDIR=$OUTDIR"
```

## Parameters

| parameter | value | rationale |
|-----------|-------|-----------|
| `--duration` | 30 | Long-form: 4500-word target @ 150wpm |
| `--limit` | 10 | 10 topics × 5 shows = 50 pieces (overnight feasible) |
| show types | all 5 | debate, news, podcast, audiobook, lecture |
| total pieces | 50 | ~6–8 hours estimated |
| fact-check | off | dry run only (no-research flag auto-set) |

## Estimated timing

Based on overnight study: 100 pieces @ 5 min avg ~75s each = 7459s total.
30-min content is ~6× longer; generation likely scales sub-linearly but conservatively:
- 50 pieces × ~450s each = ~22,500s = ~6.25 hours ✓ overnight feasible

## Monitor progress

```bash
tail -f /tmp/vox_longform_battery.log
```
or check if still running:
```bash
pgrep -la python | grep study_battery
```

## Analyze when complete

```bash
source .venv/bin/activate
python tools/analyze_battery.py study_results/longform_<timestamp> --duration 30
```

Compare against:
- `study_results/overnight_20260430_211235` (5-min, 100 pieces) — `--duration 5`
- `study_results/20260430_195715` (3-min, 25 pieces) — `--duration 3`

## What to look for in analysis

### Overshoot scaling
Does the overshoot ratio stay at ~139%/150% for debate/news or diverge?
Expected hypothesis: overshoot compounds at long duration because the conductor's
target rebalancing (`OVERSHOOT_COMPENSATION = 0.92`) is applied per-turn, not globally.

### Name diversity score
At 5-min: diversity = 0.550 (100 pieces).
At 30-min: same 50 pieces but each piece has more events = more name slots.
Hypothesis: diversity score improves because more events dilute top attractors.

### Event count vs length
At 5-min: debate avg 16.4 events, lecture avg 5.0 events.
At 30-min: does the conductor generate 6× more events (same avg length) or same
count with 6× longer events? The answer tells us whether segment budgeting works.

### Lecture coherence
Lecture at 5-min: avg 5.0 events, 853 words.
At 30-min: ~4500 words expected. Does it maintain structure, or does it repeat?
Check voice_hints (currently 0 — is the wiring bug confirmed at 30 min too?).

### Tone/pace variety at scale
Lecture at 5-min: only 2 distinct paces, 9 distinct tones.
At 30-min, with more events: do more paces emerge or does it become more monotone?

## Comparison table template (fill in after analysis)

| metric | 3-min (25 pcs) | 5-min (100 pcs) | 30-min (50 pcs) |
|--------|---------------|-----------------|-----------------|
| Name diversity | 0.721 | 0.550 | ? |
| Debate overshoot | 177% | 139% | ? |
| News overshoot | 174% | 150% | ? |
| Lecture overshoot | 108% | 114% | ? |
| Lecture nonverbal avg | 0.0 | 0.3 | ? |
| Lecture pace monotony | 0.90 | 0.53 | ? |
| Lecture voice_hints diversity | 0.000 | 0.000 | ? |
| Debate avg events/piece | ~15 | 16.4 | ? |
| Lecture avg events/piece | ~20 | 5.0 | ? |

## Related files

- `tools/study_battery.py` — battery runner (TOPICS now has 20/show; use --limit 10 for first 10)
- `tools/analyze_battery.py` — analyzer (TARGET bug fixed 2026-05-01; use --duration flag)
- `/memories/repo/vox_romana_overnight_battery.md` — findings from 5-min study
- `/memories/repo/vox_romana_next_session_plan.md` — full engineering priority queue
