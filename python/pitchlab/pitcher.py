"""One self-consistent pitcher from the published means and SDs.

Means of different variables, taken across pitchers and across studies, do
not describe a delivery anyone throws: with every joint at its mean the
ball point peaks 13 ms early and is already diving 20–60° at release. This
module looks for the single set of values, each within its published SD,
that is closest to the means (smallest sum of squared SD units) while being
consistent with how a ball is released:

- the hand stops accelerating the ball (ball-point speed peak, the push-off
  of `fullbody.push_off_time`) `PUSH_OFF_BEFORE_S` before the published
  release, because the fingers then roll it for ~8 ms (Matsuo et al. 2018)
  before it leaves;
- `LAUNCH_AFTER_PUSH_S` after push-off the ball point moves the way pitched
  balls leave the hand, `LAUNCH_VERTICAL_DEG` (Statcast four-seam release
  angle, see README) and straight toward the plate. 3 ms is where the full
  simulation's launch direction falls on this path (checked on the means and
  on the calibrated pitcher: within 1°/ms of sweep);
- the ball-point speed at push-off is the published ball speed plus the
  roll's gain (`PUSH_OFF_SPEED`).

The launch direction is a constraint taken from ball-tracking data, not from
the kinematics papers, so stage-1's release-angle check is not independent
of it; the simulation still has to reproduce it through the fingers.

Stage 2 (`--refine`) then moves a few arm and timing values against the full
simulation, so its launch, speed, and release instant land where stage 1's
proxy put them, and its loads (filtered like Fleisig's) sit within Fleisig
et al.'s (1995) kinetics of the same pitchers. With that, the stage-1 load
check measures how far the simulated pitcher is from those kinetics; it is
no longer an independent prediction.

Run `python -m pitchlab.pitcher` (then `--refine`) to recompute `single_pitcher.json`.
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

from pitchlab.literature import HEIGHT_M, MASS_KG, PITCHER_FILE, _build, _Deviations

PUSH_OFF_BEFORE_S = 0.010
LAUNCH_AFTER_PUSH_S = 0.003
LAUNCH_VERTICAL_DEG = (-2.5, 1.0)
LAUNCH_LATERAL_DEG = (0.0, 1.5)
PUSH_OFF_SPEED = (40.0, 1.0)
PEAK_TOLERANCE_M_S = 0.2
Z_LIMIT = 2.5


def proxy(z: np.ndarray, fs: float = 1000.0, fit_dt: float = 0.0005) -> dict[str, float]:
    dev = _Deviations(z=z)
    _, x = _build(dev, HEIGHT_M, MASS_KG, fs, fit_dt=fit_dt)
    t, ball, T = x["time"], x["ball"], float(x["T"])
    v = np.gradient(ball, t, axis=0)
    s = np.linalg.norm(v, axis=1)

    def at(arr: np.ndarray, when: float) -> np.ndarray:
        return np.array([np.interp(when, t, arr[:, k]) for k in range(arr.shape[1])]) if arr.ndim > 1 else np.interp(when, t, arr)

    window = (t > T - 0.04) & (t < T + 0.01)
    k = int(np.argmax(np.where(window, s, 0.0)))
    v_mid = at(v, T - PUSH_OFF_BEFORE_S + LAUNCH_AFTER_PUSH_S)
    return {
        "peak_speed": float(s[k]),
        "peak_time_ms": float((t[k] - T) * 1000.0),
        "push_speed": float(at(s, T - PUSH_OFF_BEFORE_S)),
        "vertical_deg": math.degrees(math.atan2(v_mid[2], math.hypot(v_mid[0], v_mid[1]))),
        "lateral_deg": math.degrees(math.atan2(-v_mid[0], v_mid[1])),
        "labels": dev.labels,
    }


def residuals(z: np.ndarray, anchor: np.ndarray | None = None, fit_dt: float = 0.0005) -> np.ndarray:
    """SD units from `anchor` (the published means, when anchor is 0) plus the release constraints.

    A population draw passes its own vector as `anchor`, so the fit stays near
    that pitcher instead of collapsing every draw onto the mean.
    """
    p = proxy(z, fit_dt=fit_dt)
    origin = np.zeros_like(z) if anchor is None else np.asarray(anchor, float)
    over = np.maximum(np.abs(z) - Z_LIMIT, 0.0)
    return np.concatenate([
        z - origin,
        10.0 * over,
        [
            (p["peak_speed"] - p["push_speed"]) / PEAK_TOLERANCE_M_S,
            (p["push_speed"] - PUSH_OFF_SPEED[0]) / PUSH_OFF_SPEED[1],
            (p["vertical_deg"] - LAUNCH_VERTICAL_DEG[0]) / LAUNCH_VERTICAL_DEG[1],
            (p["lateral_deg"] - LAUNCH_LATERAL_DEG[0]) / LAUNCH_LATERAL_DEG[1],
        ],
    ])


def calibrate(iterations: int = 25, step: float = 0.02, log=print, z0=None, anchor=None, fit_dt: float = 0.0005) -> np.ndarray:
    """Levenberg–Marquardt on `residuals` with a forward-difference Jacobian.

    `anchor` is the SD-unit vector the prior pulls toward. The single pitcher
    uses the means (anchor 0). Each draw of a population uses that draw.
    """
    n = len(proxy(np.zeros(64), fit_dt=fit_dt)["labels"])
    z = np.zeros(n) if z0 is None else np.array(z0, float)
    if anchor is not None:
        anchor = np.asarray(anchor, float)
    r = residuals(z, anchor, fit_dt)
    lam = 1.0
    for it in range(iterations):
        J = np.empty((len(r), n))
        for k in range(n):
            dz = z.copy()
            dz[k] += step
            J[:, k] = (residuals(dz, anchor, fit_dt) - r) / step
        g = J.T @ r
        H = J.T @ J
        while True:
            dz = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-6), -g)
            z_new = z + dz
            r_new = residuals(z_new, anchor, fit_dt)
            if r_new @ r_new < r @ r:
                z, r = z_new, r_new
                lam = max(lam / 3.0, 1e-4)
                break
            lam *= 4.0
            if lam > 1e6:
                return z
        log(f"iter {it}: cost {r @ r:.2f}  prior {np.sum(z * z):.2f}  max |z| {np.max(np.abs(z)):.2f}  "
            f"constraints {np.round(r[-4:], 2)}")
        if np.linalg.norm(dz) < 1e-3:
            break
    return z


# Stage 2 moves only these (substrings of the labels) and scores the full simulation.
REFINE = (
    "angle 179±8 @ 0.81T", "angle 165±11 @ 0.62T", "angle 105±5 @ 1.00T", "angle 18±7 @ 0.80T",
    "angle 37±6 @ 1.00T", "rate 2954±789", "rate -2438±270", "cycle", "angle 94±21 @ 0.62T", "angle 11±11 @ 0.62T",
)
SIM_SPEED = (39.6, 1.0)  # m/s, middle of stage 1's 37.5–41.7
RELEASE_TIMING_MS = 2.0
# Fleisig et al. (1995), same 26 pitchers as the kinematics: mean ± SD, N·m.
KINETICS = {"shoulder IR moment": (67.0, 11.0), "elbow varus moment": (64.0, 12.0)}


def sim_residuals(z: np.ndarray, use_kinetics: bool = True) -> tuple[np.ndarray, object]:
    import pitchlab.fullbody as fb
    from pitchlab.literature import literature_motion

    r = fb.simulate_delivery("four-seam", literature_motion(z=z))
    extra = [
        (r.launch_deg - LAUNCH_VERTICAL_DEG[0]) / LAUNCH_VERTICAL_DEG[1],
        (r.launch_horizontal_deg - LAUNCH_LATERAL_DEG[0]) / LAUNCH_LATERAL_DEG[1],
        (r.speed_m_s - SIM_SPEED[0]) / SIM_SPEED[1],
        (r.release_ms - r.obp_release_ms) / RELEASE_TIMING_MS,
    ]
    if use_kinetics:
        extra += [(r.fleisig_loads[k][0] - m) / sd for k, (m, sd) in KINETICS.items()]
    over = np.maximum(np.abs(z) - Z_LIMIT, 0.0)
    return np.concatenate([z, 10.0 * over, extra]), r


def refine(z: np.ndarray, iterations: int = 6, step: float = 0.1, log=print, use_kinetics: bool = True) -> np.ndarray:
    labels = proxy(z)["labels"]
    idx = [i for i, label in enumerate(labels) if any(label.startswith(k) for k in REFINE)]
    z = np.array(z, float)
    r, res = sim_residuals(z, use_kinetics)
    lam = 0.3
    for it in range(iterations):
        J = np.empty((len(r), len(idx)))
        for c, i in enumerate(idx):
            dz = z.copy()
            dz[i] += step
            J[:, c] = (sim_residuals(dz, use_kinetics)[0] - r) / step
        g, H = J.T @ r, J.T @ J
        for _ in range(4):
            d = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-6), -g)
            z_new = z.copy()
            z_new[idx] += d
            r_new, res_new = sim_residuals(z_new, use_kinetics)
            if r_new @ r_new < r @ r:
                z, r, res = z_new, r_new, res_new
                lam = max(lam / 3.0, 1e-3)
                break
            lam *= 4.0
        else:
            break
        log(f"refine {it}: cost {r @ r:.2f}  prior {np.sum(z * z):.2f}  launch {res.launch_deg:+.1f}/"
            f"{res.launch_horizontal_deg:+.1f}  speed {res.speed_m_s:.1f}  release {res.release_ms - res.obp_release_ms:+.1f} ms  "
            f"loads13 {res.fleisig_loads['shoulder IR moment'][0]:.0f}/{res.fleisig_loads['elbow varus moment'][0]:.0f}")
    return z


def consistent_draws(n: int, seed: int = 1, iterations: int = 8, log=print) -> list:
    """One delivery per draw, each minimized the way the single pitcher is.

    Independent N(0, 1) draws are not deliveries anybody throws: hand speed and
    launch come out of different papers and do not share a time base. Each draw
    is the prior. The same release constraints and the same |z| ≤ 2.5 box are
    then applied, so the pitcher stays near that draw while the ball point
    leaves toward the plate at a pitched speed.
    """
    from pitchlab.literature import literature_motion

    n_z = len(proxy(np.zeros(64), fit_dt=0.002)["labels"])
    rng = np.random.default_rng(seed)
    motions = []
    for k in range(n):
        z0 = rng.standard_normal(n_z)
        log(f"draw {k}: max |z| {np.max(np.abs(z0)):.2f}  sum z² {np.sum(z0 * z0):.1f}")
        z = calibrate(iterations=iterations, step=0.08, log=log, z0=z0.copy(), anchor=z0, fit_dt=0.002)
        motion = literature_motion(z=z)
        motion.id = f"literature-{seed}-{k}"
        motions.append(motion)
        p = proxy(z)
        log(f"  fine proxy  push {p['push_speed']:.1f} m/s  launch {p['vertical_deg']:+.1f}/{p['lateral_deg']:+.1f}  "
            f"peak {p['peak_time_ms']:+.1f} ms  sum z² {np.sum(z * z):.1f}")
    return motions


def main() -> None:
    t0 = time.time()
    from pitchlab.literature import single_pitcher_z

    log = lambda s: print(s, flush=True)  # noqa: E731
    if "--refine" in sys.argv:
        z = refine(single_pitcher_z(), log=log)
    else:
        z = calibrate(log=log, z0=single_pitcher_z() if "--warm" in sys.argv else None)
    p = proxy(z)
    base = proxy(np.zeros_like(z))
    stage = "refined against the simulation" if "--refine" in sys.argv else "kinematic proxy only"
    out = {
        "note": "SD-unit deviations from the published means, one per target in `labels` order. "
                "Generated by `python -m pitchlab.pitcher`.",
        "stage": stage,
        "constraints": {
            "push_off_before_release_s": PUSH_OFF_BEFORE_S,
            "launch_after_push_off_s": LAUNCH_AFTER_PUSH_S,
            "launch_vertical_deg": LAUNCH_VERTICAL_DEG,
            "launch_lateral_deg": LAUNCH_LATERAL_DEG,
            "push_off_speed_m_s": PUSH_OFF_SPEED,
            "z_limit": Z_LIMIT,
            **({"refined": list(REFINE), "sim_speed_m_s": SIM_SPEED, "kinetics": KINETICS} if "--refine" in sys.argv else {}),
        },
        "means_proxy": {k: v for k, v in base.items() if k != "labels"},
        "pitcher_proxy": {k: v for k, v in p.items() if k != "labels"},
        "z": [round(float(v), 4) for v in z],
        "labels": p["labels"],
    }
    PITCHER_FILE.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("means_proxy", "pitcher_proxy")}, indent=1))
    for label, v in sorted(zip(p["labels"], z), key=lambda a: -abs(a[1])):
        print(f"{v:+.2f}  {label}")
    print(f"{time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
