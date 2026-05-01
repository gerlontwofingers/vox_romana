"""Deep analysis of a study_battery run — surface patterns for improvement."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean

SHOW_TYPES = ["debate", "news", "podcast", "audiobook", "lecture"]


def main(study_dir: str) -> None:
    study = Path(study_dir)
    idx = json.loads((study / "index.json").read_text())
    results = idx["results"]

    def load_full(r: dict) -> dict:
        return json.loads((study / r["json_path"]).read_text())

    pieces = [(r, load_full(r)) for r in results if r.get("ok")]

    print("=" * 78)
    print(f"BATTERY ANALYSIS — {len(pieces)} pieces — {study.name}")
    print("=" * 78)

    # 1. Cast/Speaker recurrence
    print("\n## 1. NAME BIAS — characters re-used across topics")
    cast_counter: Counter[str] = Counter()
    cast_by_show: dict[str, Counter[str]] = defaultdict(Counter)
    for r, _ in pieces:
        for c in r["summary"].get("cast", []):
            cast_counter[c["name"]] += 1
            cast_by_show[r["show_type"]][c["name"]] += 1
    print(f"  Unique character names across all pieces: {len(cast_counter)}")
    print(f"  Most-reused names (>=2 appearances):")
    for name, n in cast_counter.most_common(20):
        if n >= 2:
            shows = sorted({s for s, c in cast_by_show.items() if c.get(name)})
            print(f"    {n}x  {name:38s}  ({', '.join(shows)})")

    # 2. Cast size
    print("\n## 2. CAST SIZE per show type")
    for show in SHOW_TYPES:
        sizes = [len(r["summary"].get("cast", [])) for r, _ in pieces if r["show_type"] == show]
        if sizes:
            print(f"  {show:10s}  cast_size min={min(sizes)} max={max(sizes)} mean={mean(sizes):.1f}")

    # 3. Speaker turn balance
    print("\n## 3. SPEAKER TURN BALANCE (max share of turns per piece)")
    for show in SHOW_TYPES:
        shares = []
        for r, full in pieces:
            if r["show_type"] != show:
                continue
            evs = full.get("events") or full.get("script", {}).get("events") or []
            spk = Counter(e.get("speaker_name") for e in evs if e.get("speaker_name"))
            if spk:
                total = sum(spk.values())
                shares.append(max(spk.values()) / total)
        if shares:
            print(f"  {show:10s}  dominant share mean={mean(shares):.2f} max={max(shares):.2f}")

    # 4. Nonverbal cues
    print("\n## 4. NONVERBAL CUES")
    for show in SHOW_TYPES:
        counts = [r["summary"].get("nonverbal_cue_count", 0) for r, _ in pieces if r["show_type"] == show]
        all_cues: list[str] = []
        for r, _ in pieces:
            if r["show_type"] == show:
                all_cues.extend(r["summary"].get("nonverbal_unique", []))
        if counts:
            print(f"  {show:10s}  cue_count avg={mean(counts):.1f} unique_total={len(set(all_cues))}")
            if all_cues:
                top = Counter(all_cues).most_common(8)
                print(f"             top: {[c for c,_ in top]}")

    # 5. Word count vs target
    print("\n## 5. WORD COUNT vs 3-MIN TARGET (~450 words @ 150wpm)")
    TARGET = 450
    for show in SHOW_TYPES:
        wcs = [r["summary"].get("word_count", 0) for r, _ in pieces if r["show_type"] == show]
        if wcs:
            over = sum(1 for w in wcs if w > TARGET * 1.3)
            under = sum(1 for w in wcs if w < TARGET * 0.7)
            print(f"  {show:10s}  words min={min(wcs)} max={max(wcs)} mean={mean(wcs):.0f}  over+30%={over}/{len(wcs)} under-30%={under}/{len(wcs)}")

    # 6. Event length distribution
    print("\n## 6. EVENT LENGTH (words per event)")
    for show in SHOW_TYPES:
        all_lens: list[int] = []
        for r, full in pieces:
            if r["show_type"] != show:
                continue
            evs = full.get("events") or full.get("script", {}).get("events") or []
            for e in evs:
                wc = len((e.get("text") or "").split())
                if wc > 0:
                    all_lens.append(wc)
        if all_lens:
            long_n = sum(1 for l in all_lens if l > 100)
            short_n = sum(1 for l in all_lens if l < 15)
            print(f"  {show:10s}  evt_words min={min(all_lens)} max={max(all_lens)} mean={mean(all_lens):.0f}  long(>100)={long_n} short(<15)={short_n}")

    # 7. Pace / Tone
    print("\n## 7. PACE / TONE annotations")
    for show in SHOW_TYPES:
        paces: Counter[str] = Counter()
        tones: Counter[str] = Counter()
        for r, full in pieces:
            if r["show_type"] != show:
                continue
            evs = full.get("events") or full.get("script", {}).get("events") or []
            for e in evs:
                p = e.get("pace")
                if isinstance(p, list):
                    for x in p:
                        paces[str(x)] += 1
                elif p:
                    paces[str(p)] += 1
                t = e.get("tone")
                if isinstance(t, list):
                    for x in t:
                        tones[str(x)] += 1
                elif t:
                    tones[str(t)] += 1
        print(f"  {show:10s}  paces: {dict(paces.most_common(6))}")
        print(f"             tones: {dict(tones.most_common(6))}")

    # 8. Music
    print("\n## 8. MUSIC / SFX top-level fields")
    for show in SHOW_TYPES:
        has = 0
        cue_counts = []
        for r, full in pieces:
            if r["show_type"] != show:
                continue
            for key in ("music_plan", "music", "music_cues", "soundscape", "sfx", "sound_cues"):
                if key in full:
                    has += 1
                    val = full[key]
                    if isinstance(val, list):
                        cue_counts.append(len(val))
                    elif isinstance(val, dict):
                        cue_counts.append(len(val))
                    break
        print(f"  {show:10s}  has music/sfx field: {has}/5  cue_counts={cue_counts}")

    # 9. Top-level keys (what does each show actually save?)
    print("\n## 9. TOP-LEVEL JSON KEYS (presence per show)")
    for show in SHOW_TYPES:
        keys: Counter[str] = Counter()
        for r, full in pieces:
            if r["show_type"] != show:
                continue
            for k in full.keys():
                keys[k] += 1
        print(f"  {show:10s}  {dict(keys.most_common(20))}")

    # 10. Titles
    print("\n## 10. TITLES")
    for show in SHOW_TYPES:
        titles = [r["summary"].get("title", "") for r, _ in pieces if r["show_type"] == show]
        print(f"  {show}:")
        for t in titles:
            print(f"    - {t}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "study_results/20260430_175539")
