"""How the release depends on the contact model, friction, finger strength, and roll timing.

`python -m pitchlab.sensitivity` runs the default delivery once per row,
changing one setting at a time, and prints a Markdown table.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

import pitchlab.fullbody as fb
from pitchlab.constants import FRICTION


@contextmanager
def _patched(**values):
    saved = {k: getattr(fb, k) for k in values}
    try:
        for k, v in values.items():
            setattr(fb, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(fb, k, v)


def _caps(scale: float) -> dict[str, float]:
    return {k: v * scale for k, v in fb.TORQUE_CAP_NM.items()}


def rows(grip: str = "four-seam", roll_offsets_ms=(-1.0, -0.5, 0.5, 1.0)):
    motion = fb.default_motion()
    base = fb.simulate_delivery(grip, motion)
    roll0 = base.push_off_ms / 1000.0 - fb.ROLL_BEFORE_PUSH_S
    cases = [("baseline", {}, {})]
    for solref in ("0.002 1", "0.008 1"):
        cases.append((f"contact solref {solref.split()[0]} s (0.004)", {"CONTACT_SOLREF": solref}, {}))
    for mu in (0.80, 1.10):
        cases.append((f"friction {mu:.2f} ({FRICTION})", {}, {"friction": mu}))
    for scale in (0.5, 1.5, 2.0):
        cases.append((f"finger torque cap ×{scale:g}", {"TORQUE_CAP_NM": _caps(scale)}, {}))
    cases.append(("finger torque rises over 3 ms (step)", {"ROLL_RAMP_S": 0.003}, {}))
    cases.append(("ring/pinky stay curled (extend)", {"OPEN_EXTENDS_SUPPORT": False}, {}))
    for dt in roll_offsets_ms:
        cases.append((f"roll start {dt:+.1f} ms", {}, {"roll_start_s": roll0 + dt / 1000.0}))
    yield "baseline", base
    for label, patch, kwargs in cases[1:]:
        with _patched(**patch):
            yield label, fb.simulate_delivery(grip, motion, **kwargs)


def main() -> None:
    print("| setting | speed m/s | spin rpm | axis | vertical ° | lateral ° | leaves at ms |")
    print("|---|---|---|---|---|---|---|")
    for label, r in rows(*(sys.argv[1:2] or ["four-seam"])):
        print(f"| {label} | {r.speed_m_s:.1f} | {r.spin_rpm:.0f} | {r.clock} | {r.launch_deg:+.1f} | "
              f"{r.launch_horizontal_deg:+.1f} | {r.release_ms:.1f} |", flush=True)


if __name__ == "__main__":
    main()
