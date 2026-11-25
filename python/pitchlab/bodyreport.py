"""Tables for the full-body model: practical-use criteria, literature, Statcast.

OBP is only touched by `pick_pitches` (validation, opt-in; see `obp.enabled`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pitchlab.fullbody import DeliveryResult
from pitchlab.kinetics import FLEISIG, OBP_POI
from pitchlab.motion import Motion
from pitchlab.report import REFERENCES, plate_break_inches


@dataclass(frozen=True)
class Criterion:
    name: str
    unit: str
    lo: float
    hi: float


# Agreed stage-1 targets for the four-seam at the default pitch / population mean.
CRITERIA = (
    Criterion("release angle, vertical", "°", -5.0, 0.0),
    Criterion("release angle, lateral", "°", -5.0, 5.0),
    Criterion("ball speed", "m/s", 37.5, 41.7),
    Criterion("spin rate", "rpm", 2100.0, 2500.0),
    Criterion("spin axis, minutes from 1:00", "min", -30.0, 30.0),
    Criterion("shoulder IR moment / reference", "×", 0.0, 1.5),
    Criterion("elbow varus moment / reference", "×", 0.0, 1.5),
)


def _clock_minutes(clock: str) -> float:
    h, m = (int(x) for x in clock.split(":"))
    minutes = (h % 12) * 60 + m - 60
    return (minutes + 360) % 720 - 360


def _load_reference(r: DeliveryResult, key: str) -> tuple[float, str]:
    if key in r.obp_loads:
        return r.obp_loads[key][0], "same pitch, source inverse dynamics"
    return FLEISIG[key].mean, "Fleisig 1995"


def _model_load(r: DeliveryResult, key: str) -> float:
    """Model peak processed like its reference: OBP's 20 Hz, or Fleisig's 13.4 Hz."""
    if key in r.obp_loads or key not in r.fleisig_loads:
        return r.loads[key][0]
    return r.fleisig_loads[key][0]


def criteria_values(r: DeliveryResult) -> dict[str, float]:
    return {
        "release angle, vertical": r.launch_deg,
        "release angle, lateral": r.launch_horizontal_deg,
        "ball speed": r.speed_m_s,
        "spin rate": r.spin_rpm,
        "spin axis, minutes from 1:00": _clock_minutes(r.clock),
        "shoulder IR moment / reference": _model_load(r, "shoulder IR moment") / _load_reference(r, "shoulder IR moment")[0],
        "elbow varus moment / reference": _model_load(r, "elbow varus moment") / _load_reference(r, "elbow varus moment")[0],
    }


def criteria_rows(results: list[DeliveryResult]) -> list[dict]:
    """Pass/fail per criterion on the mean over `results` (one result: that pitch)."""
    values = [criteria_values(r) for r in results]
    rows = []
    for c in CRITERIA:
        v = np.array([x[c.name] for x in values])
        mean = float(np.mean(v))
        rows.append({
            "criterion": c.name, "unit": c.unit, "lo": c.lo, "hi": c.hi, "mean": mean,
            "sd": float(np.std(v)) if len(v) > 1 else float("nan"), "n": len(v),
            "pass": bool(c.lo <= mean <= c.hi),
        })
    return rows


def format_criteria(rows: list[dict], video_note: str = "manual check of the rendered video") -> str:
    overall = "FAIL" if any(not row["pass"] for row in rows) else "PASS"
    lines = [f"overall: {overall}", f"{'criterion':32} {'target':>20} {'result':>18}  verdict"]
    for row in rows:
        sd = f" ± {row['sd']:.1f}" if not math.isnan(row["sd"]) else ""
        target = f"{row['lo']:g} to {row['hi']:g} {row['unit']}"
        result = f"{row['mean']:.1f}{sd}"
        lines.append(f"{row['criterion']:32} {target:>20} {result:>18}  {'PASS' if row['pass'] else 'FAIL'}")
    lines.append(f"{'motion looks natural on video':32} {'—':>20} {'—':>18}  {video_note}")
    return "\n".join(lines)


def independent_rows(r: DeliveryResult) -> list[dict]:
    """Checks that were not targets of the single-pitcher fit.

    Bands are the published mean ± 2 SD. A row with no published SD is reported
    with no pass/fail.
    """
    rows = []

    def add(name: str, value: float, lo: float, hi: float, unit: str, source: str) -> None:
        rows.append({
            "criterion": name, "mean": value, "lo": lo, "hi": hi, "unit": unit, "source": source,
            "sd": float("nan"), "n": 1, "pass": bool(lo <= value <= hi),
        })

    rel = r.release_ms
    ir_ms = r.fleisig_loads["shoulder IR moment"][1]
    comp = r.fleisig_loads.get("shoulder compression", r.loads["shoulder compression"])
    add("shoulder compression, 13.4 Hz", comp[0], 1090 - 2 * 110, 1090 + 2 * 110, "N",
        "Fleisig 1995, 1090 ± 110 N, not in the fit")
    add("MER lead before release", r.mer_ms - rel, -24 - 2 * 9, -24 + 2 * 9, "ms",
        "Barrentine 1998, −24 ± 9 ms")
    add("wrist-flexion peak vs release", r.peaks["wrist flexion"][1] - rel, 3 - 2 * 14, 3 + 2 * 14, "ms",
        "Barrentine 1998, +3 ± 14 ms")
    add("elbow flexion at release", r.elbow_at_release_deg, 22 - 2 * 4, 22 + 2 * 4, "°",
        "Fleisig Table 2, 22 ± 4°")
    for finger, mean, sd in (("index", 95.6, 14.5), ("middle", 89.6, 18.5)):
        parts = r.finger_force_parts.get(finger) if r.finger_force_parts else None
        if parts is not None:
            add(f"{finger} force, 1 kHz window max", parts["khz_max_n"], mean - 2 * sd, mean + 2 * sd, "N",
                f"Kinoshita 2017, {mean:g} ± {sd:g} N, max from −15 to −4 ms")
        elif finger in r.finger_force_n:
            add(f"{finger} force, last 15 ms", r.finger_force_n[finger], mean - 2 * sd, mean + 2 * sd, "N",
                f"Kinoshita 2017, {mean:g} ± {sd:g} N")
    # No published SD: reported, not judged.
    frac = ir_ms / rel if rel else float("nan")
    rows.append({
        "criterion": "IR-moment time / cycle", "mean": frac, "lo": 0.62, "hi": 0.62, "unit": "",
        "source": "Fleisig 1995, 62% of foot contact to release (no SD published)",
        "sd": float("nan"), "n": 1, "pass": False, "report_only": True,
    })
    rows.append({
        "criterion": "arm slot", "mean": r.arm_slot_deg, "lo": 37.1, "hi": 37.1, "unit": "°",
        "source": "Statcast 2024 RHP mean 37.1° (no SD published)",
        "sd": float("nan"), "n": 1, "pass": False, "report_only": True,
    })
    return rows


def format_independent(rows: list[dict]) -> str:
    lines = ["independent of the fit (not part of the stage-1 verdict)"]
    for row in rows:
        if row.get("report_only"):
            verdict = "reported"
            target = f"{row['lo']:g} {row['unit']}"
        else:
            verdict = "PASS" if row["pass"] else "FAIL"
            target = f"{row['lo']:.0f} to {row['hi']:.0f} {row['unit']}"
        lines.append(f"{row['criterion']:32} {target:>22} {row['mean']:10.2f}  {verdict}  {row['source']}")
    return "\n".join(lines)


def pick_pitches(n: int, min_mph: float = 80.0) -> list[str]:
    """The fastest right-handed pitch from each of the n fastest OBP sessions (opt-in)."""
    from pitchlab.obp import load_pitches, metadata, poi_metrics

    meta = metadata()
    poi = poi_metrics()
    ids = sorted(
        (sp for sp, row in poi.items() if row.get("p_throws") == "R" and float(meta[sp]["pitch_speed_mph"]) >= min_mph),
        key=lambda sp: -float(meta[sp]["pitch_speed_mph"]),
    )
    seen: set[str] = set()
    out = []
    for start in range(0, len(ids), 4 * n):
        batch = [sp for sp in ids[start : start + 4 * n] if sp.split("_")[0] not in seen]
        for sp, pitch in load_pitches(batch).items():
            session = sp.split("_")[0]
            # The humeral rotation is integrated from OBP's joint-angle table;
            # a few pitches have none.
            if session in seen or not np.isfinite(pitch.angles["shoulder_angle_z"]).all():
                continue
            seen.add(session)
            out.append(sp)
            if len(out) >= n:
                return out
    return out


def _sim_value(r: DeliveryResult, key: str) -> float:
    if key in r.peaks:
        return r.peaks[key][0]
    if key == "mer":
        return r.mer_deg
    if key == "trunk forward at release":
        return r.trunk_forward_at_release_deg
    if key == "trunk lateral at release":
        return r.trunk_lateral_at_release_deg
    if key in r.fleisig_loads:
        return r.fleisig_loads[key][0]
    return r.loads[key][0]


def literature_rows(r: DeliveryResult, motion: Motion) -> list[dict]:
    """Model vs Fleisig; the "obp" column is the source's own metric when it has one."""
    rows = []
    for key, pub in FLEISIG.items():
        obp_key = OBP_POI.get(key)
        obp_value = float(motion.reference_metrics.get(obp_key, float("nan"))) if obp_key else float("nan")
        if key == "shoulder compression" and "shoulder compression" in r.obp_loads:
            obp_value = r.obp_loads["shoulder compression"][0]
        ref_value = r.reference_peaks.get(key, (float("nan"), 0.0))[0]
        if key == "mer":
            ref_value = r.reference_mer_deg
        rows.append(
            {
                "quantity": key,
                "unit": pub.unit,
                "model": _sim_value(r, key),
                "reference": ref_value,
                "obp": obp_value,
                "fleisig": pub.mean,
                "fleisig_sd": pub.sd,
            }
        )
    return rows


def format_literature(rows: list[dict]) -> str:
    lines = [f"{'quantity':26} {'unit':6} {'model':>8} {'ref':>8} {'source':>8} {'Fleisig':>14}"]
    for row in rows:
        def f(v: float) -> str:
            return f"{v:8.0f}" if not math.isnan(v) else f"{'—':>8}"

        lines.append(
            f"{row['quantity']:26} {row['unit']:6} {f(row['model'])} {f(row['reference'])} {f(row['obp'])} "
            f"{row['fleisig']:7.0f} ± {row['fleisig_sd']:<4.0f}"
        )
    return "\n".join(lines)


def statcast_row(r: DeliveryResult) -> dict:
    v, w = r.release_frame()
    ivb, hb = plate_break_inches(v, w)
    ref = REFERENCES["four-seam"]
    return {
        "mph": r.speed_mph,
        "rpm": r.spin_rpm,
        "clock": r.clock,
        "eff": r.spin_efficiency,
        "ivb": ivb,
        "hb": hb,
        "ref": ref,
        "radar_mph": r.radar_mph,
    }


def format_statcast(row: dict) -> str:
    ref = row["ref"]
    return (
        f"full-body four-seam: {row['mph']:.1f} mph (this pitch's radar {row['radar_mph']:.1f}, "
        f"Statcast RHP mean {ref['mph']:.1f})  {row['rpm']:.0f} rpm ({ref['rpm']:.0f})  "
        f"axis {row['clock']} ({ref['clock']})  eff {row['eff']:.2f} ({ref['eff']:.2f})  "
        f"IVB {row['ivb']:+.1f} in ({ref['ivb']:+.1f})  HB {row['hb']:+.1f} in ({ref['hb']:+.1f})"
    )


def validation_rows(results: list[DeliveryResult]) -> list[dict]:
    rows = []
    for r in results:
        rows.append(
            {
                "pitch": r.pitch_id,
                "radar_m_s": r.radar_mph * 0.44704,
                "speed_m_s": r.speed_m_s,
                "rpm": r.spin_rpm,
                "clock": r.clock,
                "launch_v": r.launch_deg,
                "launch_h": r.launch_horizontal_deg,
                "push_minus_obp_ms": r.push_off_ms - r.obp_release_ms,
                "leaves_minus_push_ms": r.release_ms - r.push_off_ms,
                "mer": r.mer_deg,
                "ir_moment": r.loads["shoulder IR moment"][0],
                "obp_ir_moment": r.obp_loads.get("shoulder IR moment", (float("nan"), 0))[0],
                "varus": r.loads["elbow varus moment"][0],
                "obp_varus": r.obp_loads.get("elbow varus moment", (float("nan"), 0))[0],
                "sh_comp": r.loads["shoulder compression"][0],
                "obp_sh_comp": r.obp_loads.get("shoulder compression", (float("nan"), 0))[0],
                "el_comp": r.loads["elbow compression"][0],
                "obp_el_comp": r.obp_loads.get("elbow compression", (float("nan"), 0))[0],
                "tracking_rms_deg": r.tracking_rms_deg,
                "wrist_rms_mm": r.hand_error_mm,
            }
        )
    return rows


def format_validation(rows: list[dict]) -> str:
    head = (
        f"{'pitch':8} {'radar':>5} {'ball':>5} {'rpm':>5} {'axis':>5} {'launch v/h':>12} {'push':>5} {'stuck':>5} "
        f"{'MER':>4} {'IR mom':>9} {'varus':>9} {'sh comp':>11} {'el comp':>11}"
    )
    lines = [head]
    for r in rows:
        lines.append(
            f"{r['pitch']:8} {r['radar_m_s']:5.1f} {r['speed_m_s']:5.1f} {r['rpm']:5.0f} {r['clock']:>5} "
            f"{r['launch_v']:+5.1f}/{r['launch_h']:+5.1f} {r['push_minus_obp_ms']:+5.1f} {r['leaves_minus_push_ms']:+5.1f} "
            f"{r['mer']:4.0f} {r['ir_moment']:4.0f}/{r['obp_ir_moment']:4.0f} {r['varus']:4.0f}/{r['obp_varus']:4.0f} "
            f"{r['sh_comp']:5.0f}/{r['obp_sh_comp']:5.0f} {r['el_comp']:5.0f}/{r['obp_el_comp']:5.0f}"
        )

    def stat(key: str) -> str:
        v = np.array([r[key] for r in rows], dtype=float)
        return f"{np.nanmean(v):.1f} ± {np.nanstd(v):.1f}"

    def ratio(a: str, b: str) -> str:
        v = np.array([r[a] / r[b] for r in rows], dtype=float)
        return f"{np.nanmean(v):.2f} ± {np.nanstd(v):.2f}"

    lines += [
        f"n = {len(rows)}",
        f"  ball speed {stat('speed_m_s')} m/s (radar {stat('radar_m_s')})",
        f"  launch vertical {stat('launch_v')}°, toward glove side {stat('launch_h')}°",
        f"  push-off minus OBP release {stat('push_minus_obp_ms')} ms; ball leaves {stat('leaves_minus_push_ms')} ms after push-off",
        f"  model / OBP: IR moment {ratio('ir_moment', 'obp_ir_moment')}, varus {ratio('varus', 'obp_varus')}, "
        f"shoulder compression {ratio('sh_comp', 'obp_sh_comp')}, elbow compression {ratio('el_comp', 'obp_el_comp')}",
        f"  MER {stat('mer')}° (Fleisig {FLEISIG['mer'].mean:.0f} ± {FLEISIG['mer'].sd:.0f})",
        f"  tracking rms {stat('tracking_rms_deg')}°, wrist position rms {stat('wrist_rms_mm')} mm",
    ]
    return "\n".join(lines)
