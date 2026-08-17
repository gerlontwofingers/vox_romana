"""Watchdog unit test — runs in subprocess so os._exit() doesn't kill pytest."""
from __future__ import annotations

import subprocess
import sys
import time


WATCHDOG_SCRIPT = """
import time
from vox_romana.university.university import University
from vox_romana.university.config import UniversityConfig

# Trigger watchdog to fire 0.5s in the future, with 0.5s grace = 1s total.
deadline = time.monotonic() + 0.5
University._start_deadline_watchdog(deadline, grace_secs=0.5)

# Sleep way past kill point — watchdog must hard-exit us.
time.sleep(10.0)
print("FAIL: watchdog did not fire")
"""


def test_watchdog_hard_exits_process():
    """Watchdog must kill the process within (deadline + grace) seconds."""
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", WATCHDOG_SCRIPT],
        capture_output=True,
        text=True,
        timeout=8.0,
    )
    elapsed = time.monotonic() - started

    # Must NOT have completed normally (which would require sleep(10) to finish).
    assert "FAIL" not in result.stdout, f"watchdog did not fire: {result.stdout}"
    # Must have died well before the 10s sleep would complete.
    assert elapsed < 5.0, f"watchdog fired too late: {elapsed:.2f}s"
    # os._exit(1) → non-zero return code.
    assert result.returncode != 0
    # Should have printed the WATCHDOG banner.
    combined = result.stdout + result.stderr
    assert "WATCHDOG" in combined, f"missing watchdog banner: {combined!r}"


def test_watchdog_does_not_fire_before_deadline():
    """Watchdog must NOT fire if we exit before the deadline + grace."""
    script = """
import time
from vox_romana.university.university import University
deadline = time.monotonic() + 5.0
University._start_deadline_watchdog(deadline, grace_secs=5.0)
time.sleep(0.2)
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=4.0,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout
    assert "WATCHDOG" not in (result.stdout + result.stderr)
