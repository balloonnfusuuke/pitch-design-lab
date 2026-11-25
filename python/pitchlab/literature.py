"""A reference delivery from published means, as joint centers. No dataset needed.

Joint angles are minimum-jerk fits (`curvefit`) through published means and
SDs, mostly Fleisig's dissertation Table 2 (elite adults, radar 38.3 ± 0.7
m/s) with Stodden et al. (2001) for the pelvis and trunk headings and
Barrentine et al. (1998) for the wrist. Forward kinematics below turns them
into the joint centers a motion-capture system would report, using Winter's
segment fractions. Targets whose source says "assumed" are not published;
they only pin the window ends or an unreported angle.

Angle definitions (lab projections, as the papers measure them):
- pelvis and shoulder-line closure: heading of the throwing-side lateral axis
  in the horizontal plane, 90° = glove side toward home, 0° = square to home;
- trunk forward / lateral tilt: hip-center-to-thorax line projected on the
  sagittal / frontal plane, lateral toward the glove side;
- shoulder abduction, horizontal adduction, and external rotation (Fleisig's
  definition, as in `fullbody.fleisig_er_deg`) in the thorax frame;
- elbow flexion (0 = straight), wrist flexion (positive toward the palm).

`literature_motion(rng)` with a numpy Generator draws every published mean
from N(mean, SD) instead, for a population.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from pitchlab.curvefit import Target, fit
from pitchlab.motion import Motion

# de Leva (1996) male lower (OMPH–hip), middle (XYPH–OMPH), upper (SUPR–XYPH) trunk, mm.
DE_LEVA_TRUNK_M = (0.1457, 0.2155, 0.1707)

# Winter, Biomechanics and Motor Control of Human Movement, 4th ed., fractions of stature.
THIGH_FRAC = 0.245
SHANK_FRAC = 0.246
UPPER_ARM_FRAC = 0.186
FOREARM_FRAC = 0.146
TRUNK_FRAC = 0.288  # hip center to the shoulder line
HIP_WIDTH_FRAC = 0.154
SHOULDER_WIDTH_FRAC = 0.20
# Wrist to the hand's joint center (third metacarpal head region), de Leva's
# hand length 0.0862 × stature at 1.85 m is 16 cm; 9 cm is the knuckle region.
HAND_JC_M = 0.09
# Ball center from the wrist center along the hand and off the palm (ball
# radius + palm pad + PALM_GAP_M), matching the grip in `fullbody` (assumed geometry).
BALL_ALONG_HAND_M = 0.10
BALL_OFF_PALM_M = 0.062
HEIGHT_M = 1.85
MASS_KG = 88.0

# Fleisig dissertation Table 2 (foot contact = 0, release = 1 of a 0.139 ± 0.017 s interval).
CYCLE_S = (0.139, 0.017)
# Fleisig, ISBS 2010 (ojs.ub.uni-konstanz.de/cpa/article/view/4377): at front-foot
# contact the stride is 83 ± 4% of body height and the lead knee is flexed 45 ± 9°.
# Ankle-to-ankle. The dissertation Table 2 knee at foot contact is 51 ± 9° and
# that target is unchanged; 51° sits inside 45 ± 9°.
STRIDE_FRAC = 0.83
# MacWilliams, Choi, Perezous, Chao, McFarland, Am J Sports Med 26:66–71 (1998).
# Forces below are the plate's force on the body (opposite the paper's
# foot-on-plate sign). Push-off AP peaks just before foot contact; landing AP,
# vertical, and resultant peak just before release. Vertical and resultant are
# the paper's "approximately" / "about" readings of its figure, not a table.
PUSH_OFF_AP_BW = 0.35
LANDING_AP_BW = 0.72
LANDING_VERTICAL_BW = 1.5
LANDING_RESULTANT_BW = 1.75
PUSH_OFF_VERTICAL_BW = 1.0
_G = 9.80665
# How long the set and the knee lift are held. Fleisig names the phases
# (wind-up to max knee height, then stride to foot contact) but does not give
# these durations. The stride duration is not free: it is the time over which
# a push-off of PUSH_OFF_AP_BW covers the hip's travel, so the peak is the
# cited force and the clock is the consequence. 推測 for the two holds only.
SET_HOLD_S = 0.18
LIFT_S = 0.32
# After foot contact the lead hip's forward speed is brought to 0 at release
# by a half-sine. The peak is the consequence of that stop (v_fc and the
# cycle), not MacWilliams' 0.72 BW. 0.72 BW over this cycle cannot cancel
# the push-off speed. BRAKE_S remains the follow-through window after the stop.
BRAKE_S = 0.28
PELVIS_OPEN_FC = (33.0, 10.0)  # Fleisig ISBS 2010
SEPARATION_FC = (48.0, 14.0)  # same talk, shoulder line ~15° closed + pelvis 33° open
PELVIS_RATE = (640.0, 98.0, 0.29)
SHOULDER_RATE = (1233.0, 86.0, 0.50)
# Stodden et al., J Appl Biomech 17:164 (2001); SDs from Azuma et al. IJERPH 2021.
PELVIS_MER, PELVIS_BR = (85.0, 5.4), (89.0, 5.8)  # 90 = facing home
TORSO_MER, TORSO_BR = (88.0, 6.3), (111.0, 7.3)
TRUNK_FWD_FC = (4.0, 7.0)  # Fleisig ISBS 2010
TRUNK_FWD_BR = (37.0, 6.0)  # Table 2, 53 ± 6° from horizontal
TRUNK_LAT_BR = (31.0, 9.0)  # Table 2, 121 ± 9° with 90 vertical
TRUNK_FWD_RATE = (393.0, 55.0, 0.93)
MER_FRAC = 0.81
KINETIC_FRAC = 0.62  # Fleisig et al. 1995, instant of maximum shoulder IR torque
COMPRESSIVE_FRAC = 1.08
# Barrentine, Matsuo, Escamilla, Fleisig, Andrews, J Appl Biomech 14:1–23 (1998):
# fastball peak wrist flexion velocity 2954 ± 789 °/s at +3 ± 14 ms.
WRIST_RATE = (2954.0, 789.0, 0.003)
# Barrentine et al. (1998), fastball peak wrist extension 41 ± 8° (33–49).
# Our sign is flexion-positive. 44° is inside that range. 38° left the
# horseshoe grip's launch below −5°; the published mean 41° is between them.
# Barrentine's mean is 41°. 41.5° is inside 41 ± 8°. 44° was the value that
# kept launch inside the window when the wrist was a position servo. With the
# wrist on the unloaded reference torque the hand is pushed back ~9°, and
# 41.5° puts nominal launch at about −2.6°.
WRIST_EXTENSION_AT_MER_DEG = 41.5
ASSUMED = "assumed, unpublished"
F = "Fleisig dissertation Table 2"
K = "Fleisig et al. 1995 kinetics instants"
S = "Stodden et al. 2001 (SD from Azuma et al. 2021)"


class _Deviations:
    """Hands out the SD-unit deviation of each published mean, in a fixed order."""

    def __init__(self, rng: np.random.Generator | None = None, z=None):
        self.rng, self.z, self.k = rng, None if z is None else np.asarray(z, float), 0
        self.labels: list[str] = []

    def __call__(self, label: str) -> float:
        self.labels.append(label)
        k, self.k = self.k, self.k + 1
        if self.rng is not None:
            return float(self.rng.standard_normal())
        if self.z is not None and k < len(self.z):
            return float(self.z[k])
        return 0.0


def _targets(T: float, dev: _Deviations, t0: float, t1: float) -> dict[str, tuple[list[Target], float]]:
    A, R, D = "angle", "rate", "accel"

    def tg(kind: str, t: float, mean: float, sd: float, src: str) -> Target:
        if kind == D or "assumed" in src:
            return Target(kind, t, mean, sd, src)
        return Target(kind, t, mean + sd * dev(f"{kind} {mean:g}±{sd:g} @ {t / T:.2f}T ({src})"), sd, src)

    def settle(t: float, angle: float, sd: float, rate_sd: float) -> list[Target]:
        return [Target(A, t, angle, sd, ASSUMED), Target(R, t, 0.0, rate_sd, ASSUMED)]

    pelvis_fc = 90.0 - PELVIS_OPEN_FC[0]
    tr, ts = PELVIS_RATE[2] * T, SHOULDER_RATE[2] * T
    return {
        "pelvis": ([
            tg(A, 0.0, pelvis_fc, PELVIS_OPEN_FC[1], "Fleisig ISBS 2010, 33 ± 10° open"),
            tg(R, tr, -PELVIS_RATE[0], PELVIS_RATE[1], F),
            Target(D, tr, 0.0, 4000.0, F + " (time of the peak)"),
            tg(A, MER_FRAC * T, 90.0 - PELVIS_MER[0], PELVIS_MER[1], S),
            tg(A, T, 90.0 - PELVIS_BR[0], PELVIS_BR[1], S),
            Target(R, t0, 0.0, 200.0, ASSUMED),
            *settle(t1, -25.0, 15.0, 400.0),
        ], 3.0e5),
        "shoulder_line": ([
            tg(A, 0.0, pelvis_fc + SEPARATION_FC[0], SEPARATION_FC[1], "Fleisig ISBS 2010, separation"),
            tg(R, ts, -SHOULDER_RATE[0], SHOULDER_RATE[1], F),
            Target(D, ts, 0.0, 6000.0, F + " (time of the peak)"),
            tg(A, MER_FRAC * T, 90.0 - TORSO_MER[0], TORSO_MER[1], S),
            tg(A, T, 90.0 - TORSO_BR[0], TORSO_BR[1], S),
            Target(R, t0, 0.0, 200.0, ASSUMED),
            *settle(t1, -35.0, 15.0, 500.0),
        ], 6.0e5),
        "trunk_forward": ([
            tg(A, 0.0, TRUNK_FWD_FC[0], TRUNK_FWD_FC[1], "Fleisig ISBS 2010"),
            tg(A, T, TRUNK_FWD_BR[0], TRUNK_FWD_BR[1], F),
            tg(R, TRUNK_FWD_RATE[2] * T, TRUNK_FWD_RATE[0], TRUNK_FWD_RATE[1], F),
            Target(D, TRUNK_FWD_RATE[2] * T, 0.0, 3000.0, F + " (time of the peak)"),
            *settle(t0, TRUNK_FWD_FC[0], 10.0, 200.0),
            *settle(t1, TRUNK_FWD_BR[0] + 13.0, 10.0, 400.0),
        ], 2.0e5),
        "trunk_lateral": ([
            Target(A, 0.0, 0.0, 10.0, ASSUMED),
            tg(A, T, TRUNK_LAT_BR[0], TRUNK_LAT_BR[1], F),
            *settle(t0, 0.0, 10.0, 200.0),
            *settle(t1, TRUNK_LAT_BR[0] + 3.0, 10.0, 300.0),
        ], 2.0e5),
        "elbow": ([
            tg(A, 0.0, 87.0, 13.0, F),
            tg(A, 0.55 * T, 99.0, 11.0, F),
            Target(R, 0.55 * T, 0.0, 100.0, F + " (time of the maximum)"),
            tg(A, KINETIC_FRAC * T, 96.0, 10.0, K),
            tg(R, 0.91 * T, -2438.0, 270.0, F),
            Target(D, 0.91 * T, 0.0, 2.0e5, F + " (time of the peak)"),
            tg(A, T, 22.0, 4.0, F),
            tg(A, 1.03 * T, 19.0, 4.0, F),
            Target(R, 1.03 * T, 0.0, 300.0, F + " (time of the minimum)"),
            tg(A, COMPRESSIVE_FRAC * T, 25.0, 10.0, K),
            Target(A, T + 0.05, 30.0, 10.0, ASSUMED + ": the elbow stays extended through deceleration"),
            Target(A, T + 0.08, 35.0, 10.0, ASSUMED),
            *settle(t0, 95.0, 20.0, 300.0),
            *settle(t1, 30.0, 20.0, 600.0),
        ], 2.0e7),
        "external_rotation": ([
            tg(A, 0.0, 56.0, 25.0, F),
            tg(A, KINETIC_FRAC * T, 165.0, 11.0, K),
            tg(A, MER_FRAC * T, 179.0, 8.0, F),
            Target(R, MER_FRAC * T, 0.0, 300.0, F + " (time of the maximum)"),
            tg(A, T, 105.0, 5.0, "Fleisig dissertation review, 100–110° at release"),
            tg(R, 1.03 * T, -7511.0, 1140.0, F),
            Target(D, 1.03 * T, 0.0, 1.0e6, F + " (time of the peak)"),
            tg(A, COMPRESSIVE_FRAC * T, 64.0, 35.0, K),
            Target(A, T + 0.03, 15.0, 20.0, ASSUMED + ": internal rotation runs out over ~20 ms after its peak"),
            Target(A, T + 0.06, 0.0, 20.0, ASSUMED),
            Target(R, T + 0.06, 0.0, 300.0, ASSUMED),
            *settle(t0, 40.0, 30.0, 500.0),
            *settle(t1, 0.0, 30.0, 800.0),
        ], 1.5e8),
        "horizontal_adduction": ([
            tg(A, 0.0, -22.0, 14.0, F),
            tg(A, KINETIC_FRAC * T, 11.0, 11.0, K),
            tg(A, 0.80 * T, 18.0, 7.0, F),
            Target(R, 0.80 * T, 0.0, 100.0, F + " (time of the maximum)"),
            tg(A, COMPRESSIVE_FRAC * T, 6.0, 8.0, K),
            *settle(t0, -22.0, 15.0, 200.0),
            *settle(t1, -5.0, 20.0, 400.0),
        ], 1.0e6),
        "abduction": ([
            tg(A, 0.0, 91.0, 8.0, F),
            tg(A, KINETIC_FRAC * T, 94.0, 21.0, K),
            tg(A, 0.9 * T, 94.0, 9.0, F + ", mean during acceleration"),
            tg(A, COMPRESSIVE_FRAC * T, 93.0, 10.0, K),
            *settle(t0, 91.0, 10.0, 200.0),
            *settle(t1, 90.0, 10.0, 300.0),
        ], 1.0e5),
        "lead_knee": ([
            tg(A, 0.0, 51.0, 9.0, F),
            tg(A, T, 33.0, 14.0, F),
            *settle(t0, 41.0, 15.0, 300.0),
            *settle(t1, 25.0, 15.0, 300.0),
        ], 1.0e5),
        "wrist_flexion": ([
            # Inside Barrentine et al. (1998) fastball peak extension 41 ± 8°
            # (their sign; ours is flexion-positive). 41.5° is next to the
            # mean. 44° was for the position-servo wrist; 38° left launch
            # below −5° on that servo.
            Target(A, MER_FRAC * T, -WRIST_EXTENSION_AT_MER_DEG, 15.0,
                   ASSUMED + ": within Barrentine 1998 peak extension 41 ± 8°"),
            tg(R, T + WRIST_RATE[2], WRIST_RATE[0], WRIST_RATE[1], "Barrentine et al. 1998"),
            Target(D, T + WRIST_RATE[2], 0.0, 2.0e5, "Barrentine et al. 1998 (time of the peak)"),
            *settle(t0, -10.0, 20.0, 300.0),
            *settle(t1, 40.0, 20.0, 600.0),
        ], 2.0e7),
    }


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _smoother(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def _piecewise(time: np.ndarray, knots: list[tuple[float, float]]) -> np.ndarray:
    """Smoothstep through knots. Zero derivative at every knot. 推測 angles live here."""
    out = np.full(time.shape, knots[-1][1], dtype=float)
    out[time <= knots[0][0]] = knots[0][1]
    for (t0, a0), (t1, a1) in zip(knots, knots[1:]):
        m = (time >= t0) & (time <= t1)
        u = np.clip((time[m] - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
        out[m] = a0 + (a1 - a0) * _smoother(u)
    return out


# Angles at set and at maximum knee height, model convention (module docstring).
# Fleisig's ISBS text says the hands separate during the stride and names the
# phases; it does not publish these degrees. The blend joins the fitted curve
# at _FIT0, so every published target from foot contact onward is untouched.
_FIT0 = -0.12
_WINDUP_AT_SET: dict[str, float] = {
    "pelvis": 88.0,  # more closed than the 33° open at foot contact
    "shoulder_line": 92.0,
    "trunk_forward": 0.0,
    "trunk_lateral": 0.0,
    "elbow": 85.0,
    "external_rotation": 25.0,
    "horizontal_adduction": 25.0,
    "abduction": 70.0,
    "lead_knee": 20.0,
    "wrist_flexion": 0.0,
}
_WINDUP_AT_KNEE: dict[str, float] = {
    "pelvis": 82.0,
    "shoulder_line": 90.0,
    "trunk_forward": -2.0,
    "trunk_lateral": 0.0,
    "elbow": 85.0,
    "external_rotation": 25.0,
    "horizontal_adduction": 25.0,
    "abduction": 70.0,
    "lead_knee": 110.0,
    "wrist_flexion": 0.0,
}


def _hermite(time: np.ndarray, t0: float, t1: float, p0: np.ndarray, p1: np.ndarray, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    dt = max(t1 - t0, 1e-9)
    u = np.clip((time - t0) / dt, 0.0, 1.0)
    u2, u3 = u * u, u * u * u
    h00 = 2 * u3 - 3 * u2 + 1
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2
    return h00[:, None] * p0 + h10[:, None] * (dt * v0) + h01[:, None] * p1 + h11[:, None] * (dt * v1)


def _lateral(closure_deg: np.ndarray) -> np.ndarray:
    a = np.radians(closure_deg)
    return np.stack([np.cos(a), -np.sin(a), np.zeros_like(a)], axis=1)


def _knee(hip: np.ndarray, ankle: np.ndarray, bend: np.ndarray, thigh: float, shank: float) -> np.ndarray:
    reach = ankle - hip
    d = np.clip(np.linalg.norm(reach, axis=1), abs(thigh - shank) + 1e-3, thigh + shank - 1e-4)
    line = _unit(reach)
    b = bend - line * np.sum(bend * line, 1, keepdims=True)
    b = _unit(b)
    a = (thigh * thigh - shank * shank + d * d) / (2 * d)
    h = np.sqrt(np.maximum(thigh * thigh - a * a, 0.0))
    return hip + line * a[:, None] + b * h[:, None]


def _curve_deg(curves: dict, name: str, t: float) -> float:
    c = curves[name]
    grid = c.t0 + c.dt * np.arange(len(c.angle))
    return float(np.interp(t, grid, c.angle))


def _frame_axes(closure: float, fwd_deg: float, lat_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Pelvis lateral axis and superior axis at one instant."""
    fwd, lat = math.radians(fwd_deg), math.radians(lat_deg)
    sup = np.array([-math.tan(lat), math.tan(fwd), 1.0])
    sup = sup / np.linalg.norm(sup)
    h = np.array([math.cos(math.radians(closure)), -math.sin(math.radians(closure)), 0.0])
    x = h + (-float(h @ sup) / sup[2]) * np.array([0.0, 0.0, 1.0])
    return x / np.linalg.norm(x), sup


def _gap_on_the_rubber(stride: float, reach: float, dx: float, x_pel: np.ndarray, hip_half: float, maxlen: float) -> float | None:
    """Horizontal lead-hip gap that leaves the rear foot on the rubber.

    Larger gap moves both hips back toward the rubber. Returns None when even
    the largest gap the lead knee allows cannot reach the rubber.
    """
    lo, hi = 0.20, math.sqrt(max(reach * reach - dx * dx, 0.0)) * 0.96

    def rear_len(gap: float) -> float:
        lift = math.sqrt(max(reach * reach - gap * gap - dx * dx, 0.0))
        lead = np.array([-x_pel[0] * hip_half, stride - gap, lift])
        rear = lead + 2.0 * x_pel * hip_half
        return float(np.linalg.norm(rear))

    if hi <= lo or rear_len(hi) > maxlen:
        return None
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if rear_len(mid) > maxlen:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _build(dev: _Deviations, height_m: float, mass_kg: float, fs: float, fit_dt: float = 0.0005) -> tuple[Motion, dict[str, np.ndarray]]:
    T = CYCLE_S[0] + CYCLE_S[1] * dev(f"cycle {CYCLE_S[0]:g}±{CYCLE_S[1]:g} s ({F})")
    # Published angles are fit on the old window. Nothing before _FIT0 is a
    # tg() target, so the single-pitcher deviations stay aligned.
    t0, t1 = _FIT0, T + 0.12
    curves = {
        name: fit(name, targets, t0, t1, dt=fit_dt, jerk_scale=scale)
        for name, (targets, scale) in _targets(T, dev, t0, t1).items()
    }

    H = height_m
    thigh, shank = THIGH_FRAC * H, SHANK_FRAC * H
    upper, fore = UPPER_ARM_FRAC * H, FOREARM_FRAC * H
    hip_half, sh_half, trunk = 0.5 * HIP_WIDTH_FRAC * H, 0.5 * SHOULDER_WIDTH_FRAC * H, TRUNK_FRAC * H
    maxlen = 0.97 * (thigh + shank)
    knee_fc = _curve_deg(curves, "lead_knee", 0.0)
    pel_fc = _curve_deg(curves, "pelvis", 0.0)
    fwd_fc = _curve_deg(curves, "trunk_forward", 0.0)
    lat_fc = _curve_deg(curves, "trunk_lateral", 0.0)
    x_fc, _ = _frame_axes(pel_fc, fwd_fc, lat_fc)
    reach_fc = math.sqrt(thigh**2 + shank**2 + 2 * thigh * shank * math.cos(math.radians(knee_fc)))
    dx_fc = x_fc[0] * hip_half  # lead ankle x is 0; lead hip x is -x_fc[0]*hip_half
    stride = STRIDE_FRAC * H
    gap_fc = None
    for _ in range(16):
        gap_fc = _gap_on_the_rubber(stride, reach_fc, dx_fc, x_fc, hip_half, maxlen)
        if gap_fc is not None:
            break
        stride -= 0.02
    if gap_fc is None:
        raise RuntimeError("lead knee and leg length cannot span a stride")
    lift_fc = math.sqrt(max(reach_fc**2 - gap_fc**2 - dx_fc**2, 0.0))
    y_fc = stride - gap_fc
    x_knee, _ = _frame_axes(_WINDUP_AT_KNEE["pelvis"], _WINDUP_AT_KNEE["trunk_forward"], _WINDUP_AT_KNEE["trunk_lateral"])
    # Balance point: pelvis over the rubber. 推測 position, not a published coordinate.
    y_knee = 0.06 - float(x_knee[1]) * hip_half
    travel = max(y_fc - y_knee, 0.05)
    push_a = PUSH_OFF_AP_BW * _G
    stride_s = math.sqrt(travel * math.pi / push_a)
    v_fc = 2.0 * push_a * stride_s / math.pi
    t_knee = -stride_s
    t_lift = t_knee - LIFT_S
    t_set = t_lift - SET_HOLD_S

    dt = 1.0 / fs
    n_pre = int(round(-t_set / dt))
    n_post = int(round((t1 - 0.005) / dt))
    # Uniform grid with a sample at foot contact. Phase times stay where the
    # push-off integral put them; they do not have to land on a sample.
    t_set = -n_pre * dt
    time = np.concatenate([-np.arange(n_pre, 0, -1) * dt, np.arange(n_post) * dt])
    i0 = n_pre
    n = len(time)

    def ang(name: str) -> np.ndarray:
        c = curves[name]
        grid = c.t0 + c.dt * np.arange(len(c.angle))
        base = np.interp(time, grid, c.angle)
        if name not in _WINDUP_AT_SET:
            return base
        join = float(np.interp(_FIT0, grid, c.angle))
        knots = [(t_set, _WINDUP_AT_SET[name]), (t_knee, _WINDUP_AT_KNEE[name]), (_FIT0, join)]
        early = _piecewise(time, knots)
        out = base.copy()
        out[time < _FIT0] = early[time < _FIT0]
        return out

    z_up = np.tile([0.0, 0.0, 1.0], (n, 1))
    # Trunk tilt (Fleisig's hip-to-thorax line). The spine is straight
    # (assumed): the pelvis carries the same tilt as the thorax.
    fwd, lat = np.radians(ang("trunk_forward")), np.radians(ang("trunk_lateral"))
    sup = _unit(np.stack([-np.tan(lat), np.tan(fwd), np.ones(n)], 1))

    def across(closure: np.ndarray) -> np.ndarray:
        """Lateral axis ⟂ sup whose horizontal heading is the published closure."""
        h = _lateral(closure)
        return _unit(h + (-np.sum(h * sup, 1) / sup[:, 2])[:, None] * z_up)

    x_pel = across(ang("pelvis"))
    y_pel = np.cross(z_up, _lateral(ang("pelvis")))
    knee = np.radians(ang("lead_knee"))
    reach = np.sqrt(thigh**2 + shank**2 + 2 * thigh * shank * np.cos(knee))

    # After foot contact the lead foot stays planted and the lead hip's
    # forward speed falls to 0 at release, so later velocity is rotation.
    # The knee angle stays the published spline. A half-sine whose integral
    # equals v_fc over [0, T] is the brake; its peak is reported, not tuned.
    lead_hip = np.zeros((n, 3))
    lead_ankle = np.zeros((n, 3))
    lead_hip[i0:, 0] = -x_pel[i0:, 0] * hip_half
    y = np.empty(n - i0)
    vel = np.empty(n - i0)
    y[0] = y_fc
    vel[0] = v_fc
    brake_peak = v_fc * math.pi / (2.0 * max(T, 1e-3))
    for i in range(n - i0 - 1):
        t = float(time[i0 + i])
        acc = -brake_peak * math.sin(math.pi * t / T) if 0.0 <= t <= T else 0.0
        step = float(time[i0 + i + 1] - time[i0 + i])
        y[i + 1] = y[i] + vel[i] * step
        vel[i + 1] = max(0.0, vel[i] + acc * step)
        dx = float(-lead_hip[i0 + i + 1, 0])
        room = math.sqrt(max(float(reach[i0 + i + 1]) ** 2 - dx * dx, 1e-8)) * 0.96
        gap = min(max(stride - y[i + 1], 0.05), room)
        y[i + 1] = stride - gap
        if gap <= 0.05 + 1e-9 or abs(gap - room) < 1e-6:
            vel[i + 1] = 0.0
    lead_hip[i0:, 1] = y
    dx_post = -lead_hip[i0:, 0]
    gap_post = stride - lead_hip[i0:, 1]
    lift_post = np.sqrt(np.maximum(reach[i0:] ** 2 - gap_post**2 - dx_post**2, 0.0))
    lead_hip[i0:, 2] = lift_post
    lead_ankle[i0:, 1] = stride

    # Before foot contact: set, knee lift, then a stride whose hip acceleration
    # toward home peaks at the push-off shear. End state matches foot contact.
    x_set, _ = _frame_axes(_WINDUP_AT_SET["pelvis"], _WINDUP_AT_SET["trunk_forward"], _WINDUP_AT_SET["trunk_lateral"])
    y_set = 0.10 - float(x_set[1]) * hip_half
    z_set = 0.86 - float(x_set[2]) * hip_half
    z_knee = 0.92 - float(x_knee[2]) * hip_half
    pre = time < 0.0
    u_stride = np.clip((time[pre] - t_knee) / stride_s, 0.0, 1.0)
    # y(u) = y_knee + travel * (u - sin(pi u)/pi); v(1) = v_fc, a peaks at push_a.
    y_pre = np.empty(int(np.sum(pre)))
    lifting = time[pre] <= t_knee
    striding = ~lifting
    y_pre[lifting] = _hermite(time[pre][lifting], t_set, t_knee, np.array([y_set]), np.array([y_knee]), np.array([0.0]), np.array([0.0]))[:, 0]
    uu = u_stride[striding]
    y_pre[striding] = y_knee + travel * (uu - np.sin(math.pi * uu) / math.pi)
    z_end_v = float((lead_hip[i0 + 1, 2] - lead_hip[i0, 2]) / (time[i0 + 1] - time[i0]))
    z_pre = np.empty_like(y_pre)
    z_pre[lifting] = _hermite(
        time[pre][lifting], t_set, t_knee, np.array([z_set]), np.array([z_knee]), np.array([0.0]), np.array([0.0])
    )[:, 0]
    z_pre[striding] = _hermite(
        time[pre][striding], t_knee, 0.0, np.array([z_knee]), np.array([lift_fc]), np.array([0.0]), np.array([z_end_v])
    )[:, 0]
    lead_hip[pre, 0] = -x_pel[pre, 0] * hip_half
    lead_hip[pre, 1] = y_pre
    lead_hip[pre, 2] = z_pre

    ankle_h = 0.039 * H  # Winter: ankle joint above the sole
    ankle_set = np.array([-0.16, 0.18, 0.0])
    # Foot tucked up near the hip so the knee sits at hip height. The bend
    # direction is horizontal, so the knee cannot rise above the hip–ankle line.
    # Height is assumed; Fleisig names maximum knee height and does not publish it.
    ankle_knee = np.array([-0.04, 0.22, 0.84])
    ankle_fc = np.array([0.0, stride, 0.0])
    lead_ankle[pre] = _hermite(time[pre], t_set, t_knee, ankle_set, ankle_knee, np.zeros(3), np.zeros(3))
    stride_mask = pre & (time >= t_knee)
    lead_ankle[stride_mask] = _hermite(time[stride_mask], t_knee, 0.0, ankle_knee, ankle_fc, np.zeros(3), np.zeros(3))
    lead_ankle[:, 2] = np.maximum(lead_ankle[:, 2], 0.0)
    # Keep the shank inside its length while the foot is in the air.
    off = lead_ankle - lead_hip
    dist = np.linalg.norm(off, axis=1, keepdims=True)
    scale = np.minimum(1.0, (0.98 * (thigh + shank)) / np.maximum(dist, 1e-9))
    airborne = lead_ankle[:, 2] > 0.01
    lead_ankle[airborne] = (lead_hip + off * scale)[airborne]
    # Raise the whole lower body by one ankle height after the leg lengths
    # are satisfied, so the sole can sit on the mound and the knee angle stays
    # the one implied by the published flexion.
    lead_ankle[:, 2] = np.maximum(lead_ankle[:, 2], 0.0) + ankle_h
    lead_hip[:, 2] += ankle_h

    mid_hip = lead_hip + x_pel * hip_half
    rear_hip = mid_hip + x_pel * hip_half
    # Pivot foot stays on the rubber (y <= 0). The joint sits one ankle-height
    # above the mound so the sole can touch it.
    rear_ankle = np.zeros((n, 3))
    rear_ankle[:, 2] = ankle_h
    off = rear_ankle - rear_hip
    dist = np.linalg.norm(off, axis=1)
    maxlen = 0.98 * (thigh + shank)
    far = dist > maxlen
    pulled = rear_hip[far] + off[far] * (maxlen / dist[far])[:, None]
    # Never drag the pivot foot toward home. If the shortened point crosses
    # the rubber, keep y = 0 and give up x instead.
    pulled[:, 1] = np.minimum(pulled[:, 1], 0.0)
    rear_ankle[far] = pulled
    rear_ankle[:, 1] = np.minimum(rear_ankle[:, 1], 0.0)
    # The length clamp turns on in one sample and the rear knee's second
    # derivative spikes. Spread that onset over 40 ms. y stays on the rubber.
    if fs >= 100.0:
        kernel = np.ones(int(round(0.040 * fs)) | 1) 
        kernel = kernel / kernel.sum()
        for axis in (0, 2):
            rear_ankle[:, axis] = np.convolve(rear_ankle[:, axis], kernel, mode="same")
        rear_ankle[:, 1] = np.minimum(rear_ankle[:, 1], 0.0)
    # The pivot foot stays on the rubber. The 40 ms average lifts the ankle
    # about 1 cm at foot contact, and that gap is why the sole never met the
    # mound. Horizontal smoothing stays; height is the sole.
    rear_ankle[:, 2] = ankle_h
    rear_ankle[:, 1] = np.minimum(rear_ankle[:, 1], 0.0)
    lead_knee_pt = _knee(lead_hip, lead_ankle, y_pel, thigh, shank)
    rear_knee = _knee(rear_hip, rear_ankle, y_pel, thigh, shank)
    # Near full extension the triangle height's derivative blows up. The hinge
    # only needs a continuous bend plane, so the marker is averaged over 40 ms.
    if fs >= 100.0:
        kernel = np.ones(int(round(0.040 * fs)) | 1)
        kernel = kernel / kernel.sum()
        for axis in range(3):
            rear_knee[:, axis] = np.convolve(rear_knee[:, axis], kernel, mode="same")

    x_t = across(ang("shoulder_line"))
    thorax_prox = mid_hip + trunk * sup
    y_t = np.cross(sup, x_t)
    shoulder = thorax_prox + sh_half * x_t
    glove_shoulder = thorax_prox - sh_half * x_t

    # Throwing arm in the thorax frame.
    ab, ha = np.radians(ang("abduction")), np.radians(ang("horizontal_adduction"))
    d = (np.sin(ab) * np.cos(ha))[:, None] * x_t + (np.sin(ab) * np.sin(ha))[:, None] * y_t - np.cos(ab)[:, None] * sup
    e1 = _unit(y_t - d * np.sum(y_t * d, 1, keepdims=True))
    e2 = np.cross(d, e1)
    if np.mean(np.sum(e2 * sup, 1)) < 0:
        e2 = -e2
    er = np.radians(ang("external_rotation"))
    f_perp = np.cos(er)[:, None] * e1 + np.sin(er)[:, None] * e2
    el = np.radians(ang("elbow"))
    u = np.cos(el)[:, None] * d + np.sin(el)[:, None] * f_perp
    elbow = shoulder + upper * d
    wrist = elbow + fore * u
    flex_axis = _unit(np.cross(d, f_perp))

    # Hand: wrist flexion turns the hand toward the palm. No published mean
    # gives forearm pronation through the delivery, so the palm keeps one
    # angle about the forearm, the one at which it faces the wrist's travel
    # around peak wrist speed (it pushes the ball).
    b_ax = np.cross(u, flex_axis)
    v = np.gradient(wrist, time, axis=0)
    speed = np.linalg.norm(v, axis=1)
    k = int(np.argmax(np.where(time <= T, speed, 0.0)))
    win = slice(max(k - 10, 0), k + 1)
    psi = np.radians(np.arange(-180.0, 180.0, 1.0))
    va, vb = np.sum(v[win] * b_ax[win], 1), np.sum(v[win] * flex_axis[win], 1)
    psi = psi[int(np.argmax([np.sum(np.cos(p) * va + np.sin(p) * vb) for p in psi]))]
    palm = math.cos(psi) * b_ax + math.sin(psi) * flex_axis
    w = np.radians(ang("wrist_flexion"))
    hand_dir = np.cos(w)[:, None] * u + np.sin(w)[:, None] * palm
    hand = wrist + HAND_JC_M * hand_dir
    palm_normal = -np.sin(w)[:, None] * u + np.cos(w)[:, None] * palm
    ball = wrist + BALL_ALONG_HAND_M * hand_dir + BALL_OFF_PALM_M * palm_normal

    # Glove arm. Fleisig reports no glove-arm angles. The hands are together
    # until the stride and separate by foot contact (his description; the
    # directions are assumed).
    u_hand = _smoother(np.clip((time - t_knee) / (_FIT0 - t_knee), 0.0, 1.0))[:, None]
    g_d = _unit((1.0 - u_hand) * _unit(0.10 * x_t + 0.80 * y_t + 0.05 * sup) + u_hand * _unit(-0.35 * x_t + 0.55 * y_t - 0.55 * sup))
    g_u = _unit((1.0 - u_hand) * _unit(0.05 * x_t + 0.35 * y_t - 0.25 * sup) + u_hand * _unit(0.45 * x_t + 0.60 * y_t + 0.30 * sup))
    glove_elbow = glove_shoulder + upper * g_d
    glove_wrist = glove_elbow + fore * g_u

    points = {
        "thorax_prox": thorax_prox,
        "pelvis_sup": mid_hip + 0.10 * sup,
        "thorax_dist": thorax_prox - 0.20 * sup,
        "thorax_ap": thorax_prox + 0.10 * x_t,
        "shoulder_jc": shoulder,
        "elbow_jc": elbow,
        "wrist_jc": wrist,
        "hand_jc": hand,
        "elbow_lateral": elbow + 0.035 * flex_axis,
        "elbow_medial": elbow - 0.035 * flex_axis,
        "glove_shoulder_jc": glove_shoulder,
        "glove_elbow_jc": glove_elbow,
        "glove_wrist_jc": glove_wrist,
        "glove_hand_jc": glove_wrist + HAND_JC_M * g_u,
        "lead_hip": lead_hip,
        "rear_hip": rear_hip,
        "lead_knee_jc": lead_knee_pt,
        "rear_knee_jc": rear_knee,
        "lead_ankle_jc": lead_ankle,
        "rear_ankle_jc": rear_ankle,
    }
    metrics = {
        "trunk_forward_br": TRUNK_FWD_BR[0],
        "trunk_lateral_br": TRUNK_LAT_BR[0],
        "stride_m": stride,
        "stride_frac": stride / H,
        "gap_fc_m": gap_fc,
        "hip_speed_fc_m_s": v_fc,
        "stride_s": stride_s,
        "max_knee_s": t_knee,
        "brake_peak_m_s2": brake_peak,
    }
    motion = Motion(
        id="literature",
        source="literature",
        time=time,
        points=points,
        events={"fp_10_time": 0.0, "MER_time": MER_FRAC * T, "BR_time": T, "max_knee_time": t_knee},
        height_m=height_m,
        mass_kg=mass_kg,
        radar_m_s=38.3,
        reference_metrics=metrics,
    ).check()
    return motion, {"time": time, "ball": ball, "T": np.array(T)}


def _com(motion: Motion) -> np.ndarray:
    """Segment COM from de Leva fractions. The foot marker is the ankle (代用)."""
    from pitchlab.body import DE_LEVA

    p = motion.points
    t = motion.time
    n = len(t)
    acc = np.zeros((n, 3))
    mass = 0.0
    hip = 0.5 * (p["lead_hip"] + p["rear_hip"])
    trunk_vec = p["thorax_prox"] - hip
    lengths = np.array([DE_LEVA["lower_trunk"].length_m, DE_LEVA["middle_trunk"].length_m, DE_LEVA["upper_trunk"].length_m])
    lengths = lengths / lengths.sum()
    nodes = [hip]
    cursor = hip.copy()
    for frac in lengths:
        cursor = cursor + trunk_vec * frac
        nodes.append(cursor.copy())
    trunks = (
        ("lower_trunk", nodes[0], nodes[1]),
        ("middle_trunk", nodes[1], nodes[2]),
        ("upper_trunk", nodes[2], nodes[3]),
    )

    def add(frac: float, prox: np.ndarray, dist: np.ndarray, com_frac: float) -> None:
        nonlocal mass
        acc[:] += frac * (prox + (dist - prox) * com_frac)
        mass += frac

    for name, prox, dist in trunks:
        add(DE_LEVA[name].mass_frac, prox, dist, DE_LEVA[name].com_frac)
    up = _unit(trunk_vec)
    add(DE_LEVA["head"].mass_frac, p["thorax_prox"], p["thorax_prox"] + 0.12 * up, 0.5)
    limbs = (
        ("thigh", "lead_hip", "lead_knee_jc"),
        ("thigh", "rear_hip", "rear_knee_jc"),
        ("shank", "lead_knee_jc", "lead_ankle_jc"),
        ("shank", "rear_knee_jc", "rear_ankle_jc"),
        ("upper_arm", "shoulder_jc", "elbow_jc"),
        ("upper_arm", "glove_shoulder_jc", "glove_elbow_jc"),
        ("forearm", "elbow_jc", "wrist_jc"),
        ("forearm", "glove_elbow_jc", "glove_wrist_jc"),
        ("hand", "wrist_jc", "hand_jc"),
        ("hand", "glove_wrist_jc", "glove_hand_jc"),
    )
    for name, a, b in limbs:
        add(DE_LEVA[name].mass_frac, p[a], p[b], DE_LEVA[name].com_frac)
    # Feet have no marker past the ankle. Their mass sits on the ankle.
    add(DE_LEVA["foot"].mass_frac, p["lead_ankle_jc"], p["lead_ankle_jc"], 0.0)
    add(DE_LEVA["foot"].mass_frac, p["rear_ankle_jc"], p["rear_ankle_jc"], 0.0)
    return acc / mass


def chain_check(motion: Motion) -> dict[str, float]:
    """Stride, COM, and the ground-reaction residual of a kinematic delivery.

    The ground plane does not contact the feet, so this is the force the
    ground must have applied for the motion's COM: m a, with weight added on
    z. It is not a force-plate split. During the lead-foot swing the sum is
    the rear foot. After contact both feet can be down, so the sum is not
    MacWilliams' landing leg alone.
    """
    t = motion.time
    p = motion.points
    i0 = int(np.argmin(np.abs(t)))
    lead, rear = p["lead_ankle_jc"][i0], p["rear_ankle_jc"][i0]
    stride = float(np.linalg.norm((lead - rear)[:2]))
    pre = t < -0.02
    lifted = pre & (p["lead_ankle_jc"][:, 2] > 0.12)
    knee_t = float(motion.events.get("max_knee_time", -0.5))
    pushing = lifted & (t > knee_t)
    com = _com(motion)
    vel = np.gradient(com, t, axis=0)
    acc = np.gradient(vel, t, axis=0)
    kernel = np.ones(21) / 21.0
    acc_s = np.stack([np.convolve(acc[:, k], kernel, mode="same") for k in range(3)], axis=1)
    bw = _G
    fy, fz = acc_s[:, 1] / bw, (acc_s[:, 2] + _G) / bw
    resultant = np.sqrt(fy**2 + fz**2 + (acc_s[:, 0] / bw) ** 2)
    lat = p["rear_hip"][i0] - p["lead_hip"][i0]
    pelvis_open = 90.0 - math.degrees(math.atan2(-lat[1], lat[0]))
    line = p["thorax_prox"][i0] - 0.5 * (p["lead_hip"][i0] + p["rear_hip"][i0])
    trunk_fwd = math.degrees(math.atan2(line[1], line[2]))
    post = (t > 0.01) & (t < motion.release)

    def peak(mask: np.ndarray, series: np.ndarray, how: str) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.max(series[mask]) if how == "max" else np.min(series[mask]))

    return {
        "stride_m": stride,
        "stride_frac": stride / motion.height_m,
        "rear_ankle_y_fc_m": float(rear[1]),
        "lead_ankle_z_max_m": float(np.max(p["lead_ankle_jc"][pre, 2])) if np.any(pre) else float("nan"),
        "max_knee_s": float(t[pre][int(np.argmax(p["lead_ankle_jc"][pre, 2]))]) if np.any(pre) else float("nan"),
        "com_speed_fc_m_s": float(vel[i0, 1]),
        "com_speed_peak_m_s": float(np.max(vel[pushing, 1])) if np.any(pushing) else float("nan"),
        "push_off_ap_bw": peak(pushing, fy, "max"),
        "push_off_vertical_bw": peak(pushing, fz, "max"),
        "landing_ap_bw": peak(post, fy, "min"),
        "landing_vertical_bw": peak(post, fz, "max"),
        "landing_resultant_bw": peak(post, resultant, "max"),
        "pelvis_open_fc_deg": pelvis_open,
        "trunk_forward_fc_deg": trunk_fwd,
        "time_start_s": float(t[0]),
        "lead_hip_brake_g": peak(post, np.gradient(np.gradient(p["lead_hip"][:, 1], t), t) / _G, "min"),
    }


PITCHER_FILE = Path(__file__).with_name("single_pitcher.json")


def single_pitcher_z() -> np.ndarray | None:
    """SD-unit deviations of the calibrated single pitcher (`pitchlab.pitcher`), if present."""
    if not PITCHER_FILE.exists():
        return None
    return np.array(json.loads(PITCHER_FILE.read_text())["z"], float)


def literature_motion(rng: np.random.Generator | None = None, height_m: float = HEIGHT_M, mass_kg: float = MASS_KG,
                      fs: float = 1000.0, z=None, mean: bool = False) -> Motion:
    """Default: the single pitcher. `mean=True`: every published mean. `rng`: a draw. `z`: explicit deviations."""
    if rng is None and z is None and not mean:
        z = single_pitcher_z()
    motion, _ = _build(_Deviations(rng, z), height_m, mass_kg, fs)
    if rng is not None:
        motion.id = "literature-draw"
    elif mean or z is None:
        motion.id = "literature-mean"
    return motion


def literature_population(n: int, seed: int = 0) -> list[Motion]:
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        m = literature_motion(rng)
        m.id = f"literature-{seed}-{k}"
        out.append(m)
    return out
