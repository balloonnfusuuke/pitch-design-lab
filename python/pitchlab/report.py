"""Compare releases to 2024 Statcast averages, and fit a local linear surrogate."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from pitchlab.constants import BALL_MASS_KG, BALL_RADIUS_M, CONTACT_MS, FRICTION, SHOULDER_IR_DEG_S
from pitchlab.grips import GRIP_ORDER
from pitchlab.simulate import ReleaseResult, clock_of, simulate

# 2024 MLB regular season, Baseball Savant, right-handed pitchers.
# mph, rpm, active spin, and the Hawkeye axis are pitch-count-weighted means
# from the spin-direction leaderboard (minimum 100 pitches of that type).
# Tilt degrees are (180 − spin_axis) mod 360, which is the Savant clock
# (12:00 = pure backspin). Justin Verlander's 2024 four-seam row is
# axis 154.7° → 12:45, which matches that conversion.
# IVB and horizontal break are from the custom leaderboard, weighted by
# (season pitches × pitch-type percent) over every 2024 pitcher who threw it.
# break_x is Statcast's sign: catcher's right is positive, so a right-hander's
# arm-side run is negative.
# Active spin is Savant's fraction, not guaranteed identical to |ω × v̂| / |ω|.
# League-mean spin efficiency is not a column on the custom leaderboard.
REFERENCES = {
    "four-seam": {
        "mph": 94.72,
        "rpm": 2312.0,
        "tilt": 32.3,
        "eff": 0.901,
        "clock": "1:05",
        "ivb": 15.75,
        "hb": -7.54,
        "source": "Baseball Savant 2024 RHP, spin-direction (n≥100) and custom leaderboard",
    },
    "curve": {
        "mph": 80.29,
        "rpm": 2616.5,
        "tilt": 223.5,
        "eff": 0.713,
        "clock": "7:27",
        "ivb": -10.11,
        "hb": 9.38,
        "source": "Baseball Savant 2024 RHP, spin-direction (n≥100) and custom leaderboard",
    },
    "slider": {
        "mph": 86.23,
        "rpm": 2446.0,
        "tilt": 292.6,
        "eff": 0.324,
        "clock": "9:45",
        "ivb": 1.76,
        "hb": 4.56,
        "source": "Baseball Savant 2024 RHP, spin-direction (n≥100) and custom leaderboard",
    },
    "changeup": {
        "mph": 86.02,
        "rpm": 1843.7,
        "tilt": 59.1,
        "eff": 0.881,
        "clock": "1:58",
        "ivb": 4.93,
        "hb": -14.17,
        "source": "Baseball Savant 2024 RHP, spin-direction (n≥100) and custom leaderboard",
    },
}

# MLB.com Statcast glossary (Induced Vertical Break), 2024 league examples
# that the weighted table above is consistent with: four-seam about +16 in,
# curveball about −10 in.
GLOSSARY_IVB_IN = {"four-seam": 16.0, "curve": -10.0}

PLATE_M = 60.5 * 0.3048
# Same legal-midpoint ball as the contact model (balls.py). 0.145 kg was the
# previous rounding of that midpoint.
BALL_AREA_M2 = math.pi * BALL_RADIUS_M * BALL_RADIUS_M


def _angle_diff(a: float, b: float) -> float:
    return (a - b + 180) % 360 - 180


def _air_density(temp_c: float = 22.0) -> float:
    return 1.225 * (288.15 / (temp_c + 273.15))


def _accel(velocity: np.ndarray, omega: np.ndarray, rho: float, magnus: bool) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    g = np.array([0.0, 0.0, -9.80665])
    if speed < 1e-6:
        return g
    k = (rho * BALL_AREA_M2) / (2 * BALL_MASS_KG)
    cd = 0.30 + 0.10 / (1 + math.exp((speed - 36) / 5))
    acc = -k * cd * speed * velocity
    if magnus:
        vhat = velocity / speed
        trans = omega - vhat * float(np.dot(omega, vhat))
        trans_mag = float(np.linalg.norm(trans))
        if trans_mag > 1e-8:
            spin_parameter = (BALL_RADIUS_M * trans_mag) / speed
            cl = 0.336 * (1 - math.exp(-6.041 * spin_parameter))
            lift = np.cross(trans, vhat)
            lift /= np.linalg.norm(lift)
            acc = acc + lift * (k * cl * speed * speed)
    return acc + g


def plate_break_inches(velocity: np.ndarray, omega: np.ndarray) -> tuple[float, float]:
    """IVB and horizontal break in inches, same definition as the flight model.

    Spinless reference is drag and gravity only. No seam-shifted wake.
    Returns (ivb, hb) with catcher's right positive.
    """
    rho = _air_density()
    inch = 39.37007874015748

    def fly(use_magnus: bool) -> np.ndarray:
        pos = np.zeros(3)
        vel = np.array(velocity, dtype=float)
        om = np.array(omega, dtype=float)
        dt = 0.0005
        for _ in range(20000):
            if pos[1] >= PLATE_M:
                break
            k1_v = _accel(vel, om, rho, use_magnus)
            k1_p = vel
            k2_v = _accel(vel + 0.5 * dt * k1_v, om, rho, use_magnus)
            k2_p = vel + 0.5 * dt * k1_v
            k3_v = _accel(vel + 0.5 * dt * k2_v, om, rho, use_magnus)
            k3_p = vel + 0.5 * dt * k2_v
            k4_v = _accel(vel + dt * k3_v, om, rho, use_magnus)
            k4_p = vel + dt * k3_v
            pos = pos + (dt / 6) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
            vel = vel + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
            om *= math.exp(-dt / 30.0)
        return pos

    full = fly(True)
    bare = fly(False)
    ivb = (full[2] - bare[2]) * inch
    hb = (full[0] - bare[0]) * inch
    return float(ivb), float(hb)


@dataclass
class PitchError:
    grip: str
    sim_mph: float
    ref_mph: float
    sim_rpm: float
    ref_rpm: float
    sim_clock: str
    ref_clock: str
    mph_error: float
    rpm_error: float
    tilt_error_deg: float
    sim_eff: float
    ref_eff: float
    sim_ivb: float
    ref_ivb: float
    sim_hb: float
    ref_hb: float


def compare(results: dict[str, ReleaseResult] | None = None) -> list[PitchError]:
    if results is None:
        results = {grip: simulate(grip) for grip in GRIP_ORDER}
    rows = []
    for grip, result in results.items():
        ref = REFERENCES[grip]
        ivb, hb = plate_break_inches(result.velocity, result.omega)
        rows.append(
            PitchError(
                grip=grip,
                sim_mph=result.speed_mph,
                ref_mph=ref["mph"],
                sim_rpm=result.spin_rpm,
                ref_rpm=ref["rpm"],
                sim_clock=clock_of(result),
                ref_clock=ref["clock"],
                mph_error=result.speed_mph - ref["mph"],
                rpm_error=result.spin_rpm - ref["rpm"],
                tilt_error_deg=_angle_diff(result.tilt_deg, ref["tilt"]),
                sim_eff=result.spin_efficiency,
                ref_eff=ref["eff"],
                sim_ivb=ivb,
                ref_ivb=ref["ivb"],
                sim_hb=hb,
                ref_hb=ref["hb"],
            )
        )
    return rows


def format_table(rows: list[PitchError]) -> str:
    lines = [
        f"{'grip':10} {'mph':>7} {'Δ':>6} {'rpm':>7} {'Δ':>7} {'axis':>6} {'ref':>6} {'Δ°':>6} {'eff':>5} {'ivb':>6} {'Δ':>6} {'hb':>6} {'Δ':>6}"
    ]
    for row in rows:
        lines.append(
            f"{row.grip:10} {row.sim_mph:7.1f} {row.mph_error:+6.1f} "
            f"{row.sim_rpm:7.0f} {row.rpm_error:+7.0f} "
            f"{row.sim_clock:>6} {row.ref_clock:>6} {row.tilt_error_deg:+6.0f} {row.sim_eff:5.2f} "
            f"{row.sim_ivb:+6.1f} {row.sim_ivb - row.ref_ivb:+6.1f} "
            f"{row.sim_hb:+6.1f} {row.sim_hb - row.ref_hb:+6.1f}"
        )
    return "\n".join(lines)


def sweep_four_seam() -> dict:
    """One global-parameter sweep on the four-seam grip. No per-pitch knobs.

    Friction brackets the Yamaguchi 2022 MLB rosin band (no-seam CI through
    with-seam CI). Contact brackets Matsuo's 8 ± 1 ms. Internal rotation
    brackets Fleisig's 7511 ± 1140 deg/s.
    """
    frictions = (0.65, FRICTION, 1.107)
    contacts = (7.0, CONTACT_MS, 9.0)
    irs = (7511.0 - 1140.0, SHOULDER_IR_DEG_S, 7511.0 + 1140.0)
    rows = []
    for friction in frictions:
        for contact in contacts:
            for ir in irs:
                result = simulate("four-seam", friction=friction, contact_ms=contact, ir_deg_s=ir)
                rows.append(
                    {
                        "friction": friction,
                        "contactMs": contact,
                        "irDegS": ir,
                        "mph": result.speed_mph,
                        "rpm": result.spin_rpm,
                        "eff": result.spin_efficiency,
                        "tilt": result.tilt_deg,
                    }
                )
    design = np.array([[r["friction"], r["contactMs"], r["irDegS"], 1.0] for r in rows])
    nominal = np.array([FRICTION, CONTACT_MS, SHOULDER_IR_DEG_S])
    x = design[:, :3] - nominal
    rpm = np.array([r["rpm"] for r in rows])
    mph = np.array([r["mph"] for r in rows])
    rpm_coef, _, _, _ = np.linalg.lstsq(np.column_stack([x, np.ones(len(rows))]), rpm, rcond=None)
    mph_coef, _, _, _ = np.linalg.lstsq(np.column_stack([x, np.ones(len(rows))]), mph, rcond=None)
    return {
        "nominal": {"friction": FRICTION, "contactMs": CONTACT_MS, "irDegS": SHOULDER_IR_DEG_S},
        "rpmCoefficients": {
            "friction": float(rpm_coef[0]),
            "contactMs": float(rpm_coef[1]),
            "irDegS": float(rpm_coef[2]),
            "intercept": float(rpm_coef[3]),
        },
        "mphCoefficients": {
            "friction": float(mph_coef[0]),
            "contactMs": float(mph_coef[1]),
            "irDegS": float(mph_coef[2]),
            "intercept": float(mph_coef[3]),
        },
        "samples": rows,
    }


def predict(surrogate: dict, friction: float, contact_ms: float, ir_deg_s: float) -> tuple[float, float]:
    nom = surrogate["nominal"]
    dx = np.array([friction - nom["friction"], contact_ms - nom["contactMs"], ir_deg_s - nom["irDegS"]])
    rc = surrogate["rpmCoefficients"]
    mc = surrogate["mphCoefficients"]
    rpm = rc["intercept"] + rc["friction"] * dx[0] + rc["contactMs"] * dx[1] + rc["irDegS"] * dx[2]
    mph = mc["intercept"] + mc["friction"] * dx[0] + mc["contactMs"] * dx[1] + mc["irDegS"] * dx[2]
    return float(rpm), float(mph)


def dump_surrogate(path: str, surrogate: dict) -> None:
    clean = json.loads(json.dumps(surrogate, default=float))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(clean, handle, indent=2)
        handle.write("\n")


def errors_as_dicts(rows: list[PitchError]) -> list[dict]:
    return [asdict(row) for row in rows]
