"""Driveline OpenBiomechanics (OBP) pitching data.

Joint centers (`landmarks`), joint angles, and inverse-dynamics loads for 411
fastballs, 360 Hz, filtered at 20 Hz by OBP. Files are downloaded from the
OBP `dataset-v1` release on first use and cached under `python/data/obp/`
(or `$PITCHLAB_DATA`). Nothing from the dataset is committed to this repo.

License: OBP data are CC BY-NC-SA 4.0 with an extra exclusion. Anyone
employed by, associated with, or a significant shareholder of a professional
sports organization or financial analysis firm may not use the data at all.
See https://github.com/drivelineresearch/openbiomechanics.

Because of that exclusion this module is opt-in. Nothing here downloads or
reads OBP data unless PITCHLAB_OBP_ELIGIBLE=1 is set, which states that you
have read the license and that the exclusion does not apply to you. The
model itself runs from `literature` or your own capture (`motion`).

Model frame: +Y toward home, +Z up, +X to a right-hander's throwing side,
meters, t = 0 at lead-foot contact (OBP `fp_10_time`). OBP's lab is +x toward
home, +y toward first base. Left-handers are mirrored into right-handers.
"""

from __future__ import annotations

import csv
import math
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pitchlab.filters import lowpass

RELEASE_URL = "https://github.com/drivelineresearch/openbiomechanics/releases/download/dataset-v1/pitching_{name}.zip"
RAW_URL = "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/baseball_pitching/data/{path}"
CACHE = Path(os.environ.get("PITCHLAB_DATA", Path(__file__).resolve().parents[1] / "data" / "obp"))

# Fastest right-hander in the set, 1.854 m / 86.6 kg (close to the 1.85 m default),
# 94.4 mph on the radar.
DEFAULT_PITCH = "2916_4"

POINTS = (
    "thorax_prox",
    "thorax_dist",
    "thorax_ap",
    "shoulder_jc",
    "elbow_jc",
    "wrist_jc",
    "hand_jc",
    "glove_shoulder_jc",
    "glove_elbow_jc",
    "glove_wrist_jc",
    "glove_hand_jc",
    "lead_hip",
    "rear_hip",
    "lead_knee_jc",
    "rear_knee_jc",
    "lead_ankle_jc",
    "rear_ankle_jc",
)
ANGLES = ("elbow_angle_z", "wrist_angle_x", "wrist_angle_y", "shoulder_angle_z", "elbow_angle_x")
LOADS = (
    "shoulder_upper_arm_moment_z",
    "elbow_moment_y",
    "shoulder_upper_arm_force_z",
    "elbow_force_z",
)
EVENTS = ("fp_10_time", "fp_100_time", "MER_time", "BR_time", "MIR_time")


@dataclass
class Pitch:
    session_pitch: str
    time: np.ndarray
    points: dict[str, np.ndarray]
    angles: dict[str, np.ndarray]
    loads: dict[str, np.ndarray]
    load_time: np.ndarray
    events: dict[str, float]
    height_m: float
    mass_kg: float
    speed_mph: float
    throws: str
    level: str
    poi: dict[str, float] = field(default_factory=dict)

    @property
    def release(self) -> float:
        return self.events["BR_time"]

    @property
    def mer(self) -> float:
        return self.events["MER_time"]

    @property
    def speed_m_s(self) -> float:
        return self.speed_mph * 0.44704


ELIGIBLE_ENV = "PITCHLAB_OBP_ELIGIBLE"


class OBPNotEnabled(RuntimeError):
    pass


def enabled() -> bool:
    return os.environ.get(ELIGIBLE_ENV) == "1"


def require_enabled() -> None:
    if not enabled():
        raise OBPNotEnabled(
            "OBP data are opt-in. Its license (CC BY-NC-SA 4.0 plus an exclusion) bars anyone employed by or "
            "associated with a professional sports organization or a financial analysis firm, and any commercial "
            f"use. If none of that applies to you, set {ELIGIBLE_ENV}=1. Otherwise use the literature motion "
            "or your own capture."
        )


def _fetch(url: str, path: Path) -> Path:
    require_enabled()
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    tmp.replace(path)
    return path


def _table(name: str) -> Path:
    return _fetch(RELEASE_URL.format(name=name), CACHE / f"pitching_{name}.zip")


def _small(path: str) -> list[dict[str, str]]:
    local = _fetch(RAW_URL.format(path=path), CACHE / Path(path).name)
    with open(local, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metadata() -> dict[str, dict[str, str]]:
    return {row["session_pitch"]: row for row in _small("metadata.csv")}


def poi_metrics() -> dict[str, dict[str, str]]:
    return {row["session_pitch"]: row for row in _small("poi/poi_metrics.csv")}


def _rows(name: str, wanted: set[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    out: dict[str, list[list[float]]] = {}
    with zipfile.ZipFile(_table(name)) as archive:
        member = next(n for n in archive.namelist() if n.endswith(".csv"))
        with archive.open(member) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            header = next(reader)
            for line in reader:
                if line[0] in wanted:
                    out.setdefault(line[0], []).append([float(x) if x not in ("", "nan") else np.nan for x in line[1:]])
    return header[1:], {k: np.array(v) for k, v in out.items()}


def _to_model(xyz: np.ndarray, mirror: bool) -> np.ndarray:
    y = -xyz[:, 1] if mirror else xyz[:, 1]
    return np.stack([-y, xyz[:, 0], xyz[:, 2]], axis=1)


def _cache_file(session_pitch: str) -> Path:
    return CACHE / "pitches" / f"{session_pitch}.npz"


def load_pitches(session_pitches: list[str]) -> dict[str, Pitch]:
    """Pitches by OBP `session_pitch` id. Parsed tables are cached per pitch."""
    require_enabled()
    meta = metadata()
    poi = poi_metrics()
    missing = [sp for sp in session_pitches if not _cache_file(sp).exists()]
    if missing:
        wanted = set(missing)
        lm_head, lm = _rows("landmarks", wanted)
        ja_head, ja = _rows("joint_angles", wanted)
        fm_head, fm = _rows("forces_moments", wanted)
        for sp in missing:
            if sp not in lm:
                raise KeyError(f"{sp} is not in the OBP landmarks table")
            arrays = {"lm": lm[sp], "ja": ja[sp], "fm": fm.get(sp, np.zeros((0, len(fm_head))))}
            _cache_file(sp).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                _cache_file(sp), lm_head=np.array(lm_head), ja_head=np.array(ja_head), fm_head=np.array(fm_head), **arrays
            )
    out = {}
    for sp in session_pitches:
        with np.load(_cache_file(sp)) as z:
            out[sp] = _build(sp, z, meta[sp], poi.get(sp, {}))
    return out


def load_pitch(session_pitch: str = DEFAULT_PITCH) -> Pitch:
    return load_pitches([session_pitch])[session_pitch]


def _build(sp: str, z, meta: dict[str, str], poi: dict[str, str]) -> Pitch:
    lm_head = [str(h) for h in z["lm_head"]]
    ja_head = [str(h) for h in z["ja_head"]]
    fm_head = [str(h) for h in z["fm_head"]]
    lm, ja, fm = z["lm"], z["ja"], z["fm"]
    li = {h: i for i, h in enumerate(lm_head)}
    ji = {h: i for i, h in enumerate(ja_head)}
    fi = {h: i for i, h in enumerate(fm_head)}
    throws = poi.get("p_throws", "R") or "R"
    mirror = throws == "L"
    events = {e: float(lm[0, li[e]]) for e in EVENTS}
    fc = events["fp_10_time"]
    events = {e: v - fc for e, v in events.items()}
    time = lm[:, li["time"]] - fc
    points = {}
    for name in POINTS:
        k = li[f"{name}_x"]
        points[name] = _to_model(lm[:, k : k + 3], mirror)
    angles = {}
    ja_time = ja[:, ji["time"]] - fc
    for name in ANGLES:
        angles[name] = np.interp(time, ja_time, ja[:, ji[name]])
    loads = {name: fm[:, fi[name]] for name in LOADS if name in fi and len(fm)}
    load_time = fm[:, fi["time"]] - fc if len(fm) else np.zeros(0)

    def num(key: str) -> float:
        try:
            return float(poi.get(key, "nan"))
        except ValueError:
            return float("nan")

    return Pitch(
        session_pitch=sp,
        time=time,
        points=points,
        angles=angles,
        loads=loads,
        load_time=load_time,
        events=events,
        height_m=float(meta["session_height_m"]),
        mass_kg=float(meta["session_mass_kg"]),
        speed_mph=float(meta["pitch_speed_mph"]),
        throws=throws,
        level=meta.get("playing_level", ""),
        poi={k: num(k) for k in poi if k not in ("session_pitch", "session", "p_throws", "pitch_type")},
    )


def right_handed_pitch_ids(min_mph: float = 0.0) -> list[str]:
    meta = metadata()
    poi = poi_metrics()
    ids = [sp for sp, row in poi.items() if row.get("p_throws") == "R" and float(meta[sp]["pitch_speed_mph"]) >= min_mph]
    return sorted(ids, key=lambda sp: -float(meta[sp]["pitch_speed_mph"]))


# ---------------------------------------------------------------- raw arm markers

# OBP's elbow_jc and wrist_jc are the epicondyle and styloid marker midpoints
# filtered at 20 Hz (they agree to 2 mm). Winter residual analysis of the raw
# finger, wrist, and epicondyle markers puts the cutoff at 35–40 Hz; at 20 Hz
# the finger marker's peak speed drops by a factor 1.22 ± 0.04 (27 pitches).
ARM_CUTOFF_HZ = 40.0
ARM_MARKERS = ("ELB", "MELB", "WRA", "WRB", "FIN")


def _c3d_file(pitch: Pitch, names: list[str]) -> list[str]:
    session = pitch.session_pitch.split("_")[0].zfill(6)
    tag = f"_{round(pitch.speed_mph * 10)}.c3d"
    return [n for n in names if f"_{session}_" in n and n.endswith(tag)]


def _read_c3d(raw: bytes, labels_wanted: list[str]) -> dict[str, np.ndarray] | None:
    import warnings

    import c3d

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader = c3d.Reader(io.BytesIO(raw))
        labels = [label.strip() for label in reader.point_labels]
        if not all(label in labels for label in labels_wanted):
            return None
        frames = np.array([pts[:, :4] for _, pts, _ in reader.read_frames()])
    out = {}
    for label in labels_wanted:
        xyz = frames[:, labels.index(label), :3].copy()
        ok = frames[:, labels.index(label), 3] >= 0
        if ok.mean() < 0.95:
            return None
        if not ok.all():
            idx = np.arange(len(xyz))
            for k in range(3):
                xyz[~ok, k] = np.interp(idx[~ok], idx[ok], xyz[ok, k])
        out[label] = xyz
    return out


def arm_markers(pitch: Pitch) -> dict[str, np.ndarray] | None:
    """Raw throwing-arm markers (model frame, m) from OBP's c3d files, or None.

    The trial is the session's c3d whose radar speed matches; ties go to the
    one whose 20 Hz elbow midpoint matches OBP's elbow_jc.
    """
    cache = CACHE / "pitches" / f"{pitch.session_pitch}_arm.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files} if z.files else None
    side = "L" if pitch.throws == "L" else "R"
    labels = [side + m for m in ARM_MARKERS]
    best, best_err = None, math.inf
    with zipfile.ZipFile(_table("c3d")) as archive:
        for name in _c3d_file(pitch, archive.namelist()):
            raw = _read_c3d(archive.read(name), labels)
            if raw is None or len(raw[labels[0]]) != len(pitch.time):
                continue
            m = {k[1:]: _to_model(v, side == "L") for k, v in raw.items()}
            mid = 0.5 * (m["ELB"] + m["MELB"])
            err = float(np.sqrt(np.mean((np.stack([lowpass(mid[:, k], 360.0) for k in range(3)], 1) - pitch.points["elbow_jc"]) ** 2)))
            if err < best_err:
                best, best_err = m, err
    if best is not None and best_err > 0.01:
        best = None
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **(best or {}))
    return best


def with_marker_arm(pitch: Pitch, fc: float = ARM_CUTOFF_HZ) -> Pitch:
    """The pitch with elbow, wrist, and hand centers rebuilt from raw markers at `fc`.

    Adds `elbow_lateral` and `elbow_medial` so the humerus's axial rotation can
    be read from the epicondyle axis. hand_jc keeps OBP's definition: a fixed
    point in the frame of the wrist markers and the finger marker, located by
    least squares against OBP's 20 Hz hand_jc. Returns the pitch unchanged if
    the c3d trial is missing.
    """
    m = arm_markers(pitch)
    if m is None:
        return pitch
    fs = 1.0 / float(np.median(np.diff(pitch.time)))

    def filt(x: np.ndarray, cutoff: float) -> np.ndarray:
        return np.stack([lowpass(x[:, k], fs, cutoff) for k in range(3)], 1)

    def hand_frame(wra, wrb, fin):
        w = 0.5 * (wra + wrb)
        a = fin - w
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = wra - wrb
        b -= a * np.sum(a * b, 1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        return w, np.stack([a, b, np.cross(a, b)], axis=2)

    w20, R20 = hand_frame(*(filt(m[k], 20.0) for k in ("WRA", "WRB", "FIN")))
    local = np.mean(np.einsum("nji,nj->ni", R20, pitch.points["hand_jc"] - w20), axis=0)
    f = {k: filt(v, fc) for k, v in m.items()}
    w, R = hand_frame(f["WRA"], f["WRB"], f["FIN"])
    points = dict(pitch.points)
    points["elbow_jc"] = 0.5 * (f["ELB"] + f["MELB"])
    points["wrist_jc"] = w
    points["hand_jc"] = w + R @ local
    points["elbow_lateral"] = f["ELB"]
    points["elbow_medial"] = f["MELB"]
    return Pitch(**{**pitch.__dict__, "points": points})


# ---------------------------------------------------------------- as a Motion

LOAD_NAMES = {
    "shoulder IR moment": "shoulder_upper_arm_moment_z",
    "elbow varus moment": "elbow_moment_y",
    "shoulder compression": "shoulder_upper_arm_force_z",
    "elbow compression": "elbow_force_z",
}


def load_peaks(pitch: Pitch, after_release_s: float = 0.05) -> dict[str, tuple[float, float]]:
    """OBP's own inverse-dynamics peaks for the pitch, foot contact to 50 ms after release."""
    t = pitch.load_time
    out = {}
    for label, key in LOAD_NAMES.items():
        if key not in pitch.loads or not len(t):
            continue
        w = (t >= 0.0) & (t <= pitch.release + after_release_s)
        v = np.where(w, pitch.loads[key], -np.inf)
        k = int(np.nanargmax(v))
        out[label] = (float(v[k]), float(t[k] * 1000.0))
    return out


def to_motion(pitch: Pitch, marker_arm: bool = True):
    """The pitch as a `motion.Motion`, throwing arm rebuilt from raw markers when available."""
    from pitchlab.imitate import PRONATION_ZERO_OFFSET_DEG
    from pitchlab.motion import Motion

    p = with_marker_arm(pitch) if marker_arm else pitch
    return Motion(
        id=pitch.session_pitch,
        source="obp",
        time=p.time,
        points=p.points,
        events=p.events,
        height_m=p.height_m,
        mass_kg=p.mass_kg,
        radar_m_s=p.speed_m_s,
        pronation_deg=p.angles.get("elbow_angle_z"),
        pronation_offset_deg=PRONATION_ZERO_OFFSET_DEG,
        humeral_rotation_deg=p.angles.get("shoulder_angle_z"),
        reference_loads=load_peaks(pitch),
        reference_metrics={k: v for k, v in p.poi.items() if isinstance(v, float)},
    )
