"""Reference motion for the full-body model, independent of where it came from.

A `Motion` is joint-center trajectories in the model frame (+Y toward home,
+Z up, +X to a right-hander's throwing side, meters) with t = 0 at lead-foot
contact. Point names are those of `POINTS`. Sources:

- `literature.literature_motion()`: published means (Fleisig, Stodden,
  Barrentine), the default. No dataset is downloaded.
- `load_csv(path)`: your own capture, already reduced to joint centers.
- `mocap.landmarks_from_mediapipe`: MediaPipe Pose, via `from_points`.
- `obp.to_motion`: Driveline OpenBiomechanics, opt-in only (see `obp.py` for
  the license, which excludes anyone affiliated with a professional sports
  organization).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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
# Optional: lateral and medial epicondyle markers. With them the humerus's
# axial rotation near release comes from the epicondyle axis.
OPTIONAL_POINTS = ("elbow_lateral", "elbow_medial", "pelvis_sup")


@dataclass
class Motion:
    id: str
    source: str  # "literature", "csv", "mediapipe", "obp"
    time: np.ndarray
    points: dict[str, np.ndarray]
    events: dict[str, float]  # at least MER_time and BR_time; fp_10_time is 0
    height_m: float
    mass_kg: float
    radar_m_s: float = float("nan")
    # Forearm pronation in the source's convention, and the offset to the
    # model's (palm on the flexion side at 0). With no measurement the offset
    # is solved from the palm being behind the ball at push-off.
    pronation_deg: np.ndarray | None = None
    pronation_offset_deg: float | None = None
    # A humeral axial angle in degrees, used only without epicondyle points.
    humeral_rotation_deg: np.ndarray | None = None
    # Same-pitch reference values from the source, for comparison only:
    # load name -> (peak, ms), and kinematic metric name -> value.
    reference_loads: dict[str, tuple[float, float]] = field(default_factory=dict)
    reference_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def release(self) -> float:
        return self.events["BR_time"]

    @property
    def mer(self) -> float:
        return self.events["MER_time"]

    def check(self) -> "Motion":
        missing = [name for name in POINTS if name not in self.points]
        if missing:
            raise ValueError(f"motion {self.id} is missing points: {', '.join(missing)}")
        n = len(self.time)
        for name, xyz in self.points.items():
            if xyz.shape != (n, 3) or not np.isfinite(xyz).all():
                raise ValueError(f"point {name} must be finite and shaped ({n}, 3)")
        for e in ("MER_time", "BR_time"):
            if e not in self.events:
                raise ValueError(f"motion {self.id} needs event {e}")
        if not (self.time[0] < -0.09 and self.time[-1] > self.release + 0.08):
            raise ValueError("motion must cover at least 90 ms before foot contact to 80 ms after release")
        return self


def from_points(
    points: dict[str, np.ndarray],
    time: np.ndarray,
    mer_s: float,
    release_s: float,
    height_m: float,
    mass_kg: float,
    id: str = "capture",
    source: str = "capture",
    radar_m_s: float = float("nan"),
) -> Motion:
    """A motion from joint centers already in the model frame, t = 0 at foot contact."""
    return Motion(
        id=id,
        source=source,
        time=np.asarray(time, float),
        points={k: np.asarray(v, float) for k, v in points.items()},
        events={"fp_10_time": 0.0, "MER_time": float(mer_s), "BR_time": float(release_s)},
        height_m=height_m,
        mass_kg=mass_kg,
        radar_m_s=radar_m_s,
    ).check()


def load_csv(
    path: str | Path,
    height_m: float,
    mass_kg: float,
    mer_s: float | None = None,
    release_s: float | None = None,
    radar_m_s: float = float("nan"),
    left_handed: bool = False,
) -> Motion:
    """Joint centers from a CSV: `time` (s, 0 at foot contact) and `<point>_x/_y/_z` (m, model frame).

    Optional columns: `pronation_deg` (model convention, palm on the flexion
    side at 0) and the epicondyle points. Events come from the arguments or
    from `#MER_time=` / `#BR_time=` comment lines at the top. Left-handers
    are mirrored (x → −x) when `left_handed`.
    """
    path = Path(path)
    events: dict[str, float] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    body = []
    for line in lines:
        if line.startswith("#"):
            key, _, value = line[1:].partition("=")
            events[key.strip()] = float(value)
        elif line.strip():
            body.append(line)
    reader = csv.DictReader(body)
    rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    time = np.array([float(r["time"]) for r in rows])
    sign = -1.0 if left_handed else 1.0
    points = {}
    for name in (*POINTS, *OPTIONAL_POINTS):
        cols = [f"{name}_{a}" for a in "xyz"]
        if all(c in rows[0] for c in cols):
            xyz = np.array([[float(r[c]) for c in cols] for r in rows])
            xyz[:, 0] *= sign
            points[name] = xyz
    mer = mer_s if mer_s is not None else events.get("MER_time")
    rel = release_s if release_s is not None else events.get("BR_time")
    if mer is None or rel is None:
        raise ValueError("give mer_s and release_s, or #MER_time= and #BR_time= lines")
    motion = from_points(points, time, mer, rel, height_m, mass_kg, id=path.stem, source="csv", radar_m_s=radar_m_s)
    if "pronation_deg" in rows[0]:
        motion.pronation_deg = np.array([float(r["pronation_deg"]) for r in rows])
        motion.pronation_offset_deg = 0.0
    return motion


def save_csv(motion: Motion, path: str | Path) -> None:
    """Write a motion in the `load_csv` format (a template for your own data)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [n for n in (*POINTS, *OPTIONAL_POINTS) if n in motion.points]
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"#MER_time={motion.mer:.6f}\n#BR_time={motion.release:.6f}\n")
        w = csv.writer(f)
        w.writerow(["time", *[f"{n}_{a}" for n in names for a in "xyz"]])
        for i, t in enumerate(motion.time):
            w.writerow([f"{t:.6f}", *[f"{motion.points[n][i, k]:.5f}" for n in names for k in range(3)]])


def nan_float(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")
