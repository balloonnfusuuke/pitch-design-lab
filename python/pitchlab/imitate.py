"""Joint-center motion capture to model joint trajectories.

Input is a set of joint centers in the model frame (`obp.Pitch.points`, or
`mocap.landmarks_from_mediapipe`). Each segment's orientation is rebuilt from
those points, then expressed relative to its parent:

- thorax: long axis thorax_dist → thorax_prox, lateral axis toward thorax_ap
  (a thorax-fixed point 10 cm to the throwing side in OBP).
- pelvis: hip line plus vertical. Pelvis tilt is not in the landmarks.
- spine: the thorax-to-level-pelvis rotation is split. The swing of the long
  axis is shared equally by pelvis tilt, the lumbar joint, and the thoracic
  joint; axial twist is 25% lumbar, 75% thoracic. Both splits are assumed,
  in the direction of the published ranges of motion (lumbar axial rotation
  is small), not fitted.
- shoulder girdle: the glenohumeral center moves on a sphere about a pivot
  fixed in the thorax. The pivot is fitted per pitch (functional joint center),
  and the girdle has two hinges (protraction, elevation).
- humerus: long axis, with axial rotation from the elbow flexion plane while
  the elbow is bent; near full extension the plane is undefined, and the
  axial angle is integrated from OBP's humeral rotation rate.
- forearm pronation: OBP's elbow pronation angle plus one global offset
  between OBP's zero and the model's (PRONATION_ZERO_OFFSET_DEG). The hand's
  long axis comes from hand_jc; wrist flexion and deviation follow from it.
- legs: two-link IK onto the measured ankle with the model's lengths, knee in
  the measured knee plane, feet kept level.

Angles are splined (natural cubic) on the 360 Hz samples, then sampled on the
simulation grid with first and second derivatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pitchlab.body import BodyParams

HINGES = (
    "lead_knee",
    "lead_ankle",
    "trail_knee",
    "trail_ankle",
    "sc_prot",
    "sc_elev",
    "elbow",
    "pronation",
    "wrist_flex",
    "wrist_dev",
    "glove_sc_prot",
    "glove_sc_elev",
    "glove_elbow",
)
BALLS = ("lead_hip", "trail_hip", "lumbar", "thoracic", "gh", "glove_gh")
PELVIS_SWING_SHARE = 1.0 / 3.0
LUMBAR_SWING_SHARE = 1.0 / 3.0
LUMBAR_TWIST_SHARE = 0.25
# OBP elbow_angle_z minus the model's pronation angle (palm on the flexion
# side at 0). Calibrated by `fullbody.palm_alignment`: at push-off, when the
# hand stops accelerating the ball, the palm should be behind the ball. Over
# 29 OBP right-handers (one pitch per session, humerus from the epicondyle
# markers) the per-pitch best is median 140°, IQR 125–150° (1574_1 at 210°
# is the outlier), with median palm·velocity 0.86. One global number.
PRONATION_ZERO_OFFSET_DEG = 140.0


# ---------------------------------------------------------------- rotations


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _frame(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.stack([x, y, z], axis=-1)


def _rot(axis: str, angle: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    o, l = np.zeros_like(angle), np.ones_like(angle)
    if axis == "x":
        rows = [[l, o, o], [o, c, -s], [o, s, c]]
    elif axis == "y":
        rows = [[c, o, s], [o, l, o], [-s, o, c]]
    else:
        rows = [[c, -s, o], [s, c, o], [o, o, l]]
    return np.moveaxis(np.array(rows), (0, 1), (-2, -1))


def _axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    axis = _unit(axis)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    rows = [
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]
    return np.moveaxis(np.array(rows), (0, 1), (-2, -1))


def quat_from_mat(R: np.ndarray) -> np.ndarray:
    """(…,3,3) → (…,4) as w, x, y, z."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.zeros(m.shape[:-2] + (4,))
    cands = np.stack(
        [
            np.stack([1 + t, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1),
            np.stack([m[..., 2, 1] - m[..., 1, 2], 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]], -1),
            np.stack([m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0], 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], m[..., 1, 2] + m[..., 2, 1]], -1),
            np.stack([m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]], -1),
        ],
        axis=-2,
    )
    diag = np.stack([t, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1)
    pick = np.argmax(diag, axis=-1)
    q = np.take_along_axis(cands, pick[..., None, None].repeat(4, -1), axis=-2)[..., 0, :]
    return _unit(q)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        -1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_log(q: np.ndarray) -> np.ndarray:
    """Rotation vector of a unit quaternion (shortest arc)."""
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1)
    angle = 2.0 * np.arctan2(s, q[..., 0])
    scale = np.where(s > 1e-12, angle / np.maximum(s, 1e-12), 2.0)
    return v * scale[..., None]


def _continuous(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    for i in range(1, len(q)):
        if float(q[i] @ q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def _swing_twist(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R = S(axis, angle) · Rz(twist): swing of the z axis, then twist about it."""
    z = R[..., :, 2]
    ez = np.array([0.0, 0.0, 1.0])
    axis = np.cross(np.broadcast_to(ez, z.shape), z)
    n = np.linalg.norm(axis, axis=-1)
    axis = np.where(n[..., None] > 1e-9, axis / np.maximum(n, 1e-12)[..., None], np.array([1.0, 0.0, 0.0]))
    angle = np.arccos(np.clip(z[..., 2], -1.0, 1.0))
    S = _axis_angle(axis, angle)
    T = np.swapaxes(S, -1, -2) @ R
    twist = np.arctan2(T[..., 1, 0], T[..., 0, 0])
    return axis, angle, twist


# ---------------------------------------------------------------- spline


class Spline:
    """Natural cubic spline through (x, Y), Y shaped (n, k)."""

    def __init__(self, x: np.ndarray, Y: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        n = len(x)
        h = np.diff(x)
        A = np.zeros((n, n))
        B = np.zeros((n, Y.shape[1]))
        A[0, 0] = A[-1, -1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            B[i] = 6.0 * ((Y[i + 1] - Y[i]) / h[i] - (Y[i] - Y[i - 1]) / h[i - 1])
        self.x, self.Y, self.h = x, Y, h
        self.M = np.linalg.solve(A, B)

    def __call__(self, xq: np.ndarray, der: int = 0) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        i = np.clip(np.searchsorted(self.x, xq) - 1, 0, len(self.x) - 2)
        h = self.h[i][:, None]
        a = (self.x[i + 1][:, None] - xq[:, None]) / h
        b = (xq[:, None] - self.x[i][:, None]) / h
        y0, y1, m0, m1 = self.Y[i], self.Y[i + 1], self.M[i], self.M[i + 1]
        if der == 0:
            return a * y0 + b * y1 + ((a**3 - a) * m0 + (b**3 - b) * m1) * h * h / 6.0
        if der == 1:
            return (y1 - y0) / h + ((1 - 3 * a * a) * m0 + (3 * b * b - 1) * m1) * h / 6.0
        return a * m0 + b * m1


# ---------------------------------------------------------------- geometry


@dataclass
class Geometry:
    """Model dimensions that come from the pitch, not from de Leva."""

    clavicle_pivot: np.ndarray  # thorax frame, from thorax_prox, meters (scaled)
    clavicle_m: float
    glove_pivot: np.ndarray
    glove_clavicle_m: float
    rubber_z: float  # sole height on the rubber above the landing area
    scale: float  # model height / subject height
    # Subject's joint-center distances (medians, scaled): upper_arm, forearm,
    # glove_upper_arm, glove_forearm, trunk (mid-hip to thorax_prox), thigh, shank,
    # hip_half; middle_trunk and upper_trunk from the trunk fit.
    lengths: dict[str, float] = field(default_factory=dict)
    lumbar_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))  # pelvis frame, from mid-hip


@dataclass
class Reference:
    t: np.ndarray
    hinge: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ball: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]  # quat, ω (child frame), α
    pelvis: dict[str, np.ndarray]  # pos, vel, acc (world); quat, w, a (body frame)
    events: dict[str, float]
    geometry: Geometry
    points: dict[str, np.ndarray]  # scaled landmarks on the grid
    pronation_offset_deg: float
    palm_home_at_release: float
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def release(self) -> float:
        return self.events["BR_time"]


def _sphere_fit(v: np.ndarray) -> tuple[np.ndarray, float]:
    A = np.hstack([2.0 * v, np.ones((len(v), 1))])
    b = np.sum(v * v, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = math.sqrt(max(1e-6, sol[3] + float(c @ c)))
    return c, r


def _thorax(p: dict[str, np.ndarray]) -> np.ndarray:
    z = _unit(p["thorax_prox"] - p["thorax_dist"])
    x = p["thorax_ap"] - p["thorax_prox"]
    x = _unit(x - z * np.sum(x * z, -1, keepdims=True))
    return _frame(x, np.cross(z, x), z)


def _level_pelvis(p: dict[str, np.ndarray]) -> np.ndarray:
    """Pelvis frame from the hips; tilted toward `pelvis_sup` when the motion has it, else level."""
    x = _unit(p["rear_hip"] - p["lead_hip"])
    if "pelvis_sup" in p:
        up = p["pelvis_sup"] - 0.5 * (p["lead_hip"] + p["rear_hip"])
    else:
        up = np.broadcast_to(np.array([0.0, 0.0, 1.0]), x.shape)
    z = _unit(up - x * np.sum(up * x, -1, keepdims=True))
    return _frame(x, np.cross(z, x), z)


def _limb(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upper-arm frame (z proximal, y = flexion side) and elbow flexion."""
    d = _unit(elbow - shoulder)
    f = _unit(wrist - elbow)
    perp = f - d * np.sum(f * d, -1, keepdims=True)
    y = np.zeros_like(perp)
    prev = None
    for i in range(len(perp)):
        n = float(np.linalg.norm(perp[i]))
        cand = perp[i] / n if n > 1e-9 else (prev if prev is not None else np.array([0.0, 1.0, 0.0]))
        if prev is not None and n < 0.2:
            w = n / 0.2
            cand = w * cand + (1 - w) * (prev - d[i] * float(prev @ d[i]))
        cand = cand - d[i] * float(cand @ d[i])
        y[i] = cand / np.linalg.norm(cand)
        prev = y[i]
    z = -d
    flex = np.arctan2(np.linalg.norm(perp, axis=-1), np.sum(f * d, -1))
    return _frame(np.cross(y, z), y, z), flex


def _humerus_blended(
    R_thorax: np.ndarray,
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    er_rate: np.ndarray | None,
    ts: np.ndarray,
    epicondyle_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper-arm frame whose axial rotation comes from the elbow plane when it is defined.

    The plane through shoulder, elbow, and wrist fixes humeral axial rotation
    only while the elbow is bent; its error grows like 1/sin(flexion), and the
    arm is nearly straight at release. Where sin(flexion) is small the axial
    angle comes from the epicondyle axis (lateral − medial marker, with the
    constant carrying offset to the elbow plane removed), or, without
    markers, is integrated from OBP's 20 Hz humeral rotation rate.
    Returns the frame, elbow flexion, and the plane weight.
    """
    R_plane, flex = _limb(shoulder, elbow, wrist)
    Rt = np.swapaxes(R_thorax, -1, -2)
    z = np.einsum("nij,nj->ni", Rt, _unit(shoulder - elbow))
    y_plane = np.einsum("nij,nj->ni", Rt, R_plane[:, :, 1])
    n = len(ts)
    y_min = np.zeros((n, 3))
    seed = y_plane[0] - z[0] * float(y_plane[0] @ z[0])
    y_min[0] = seed / np.linalg.norm(seed)
    for i in range(1, n):
        c = y_min[i - 1] - z[i] * float(y_min[i - 1] @ z[i])
        y_min[i] = c / np.linalg.norm(c)
    x_min = np.cross(y_min, z)
    phi_plane = np.unwrap(np.arctan2(-np.sum(y_plane * x_min, -1), np.sum(y_plane * y_min, -1)))
    w = np.clip((np.sin(flex) - 0.35) / 0.30, 0.0, 1.0)
    phi = np.zeros(n)
    if epicondyle_axis is not None:
        # The flexion axis is the humerus x axis, x = sin(phi) y_min + cos(phi) x_min.
        e = np.einsum("nij,nj->ni", Rt, epicondyle_axis)
        phi_epi = np.unwrap(np.arctan2(np.sum(e * y_min, -1), np.sum(e * x_min, -1)))
        bent = w >= 1.0 if np.any(w >= 1.0) else np.sin(flex) >= np.max(np.sin(flex)) - 0.05
        d = phi_plane[bent] - phi_epi[bent]
        carry = math.atan2(float(np.mean(np.sin(d))), float(np.mean(np.cos(d))))
        phi_epi = phi_epi + carry
        phi_epi += 2 * math.pi * np.round((phi_plane[bent][0] - phi_epi[bent][0]) / (2 * math.pi))
        phi = w * phi_plane + (1.0 - w) * phi_epi
    else:
        phi[0] = phi_plane[0]
        for i in range(1, n):
            dt = ts[i] - ts[i - 1]
            phi[i] = w[i] * phi_plane[i] + (1.0 - w[i]) * (phi[i - 1] - dt * er_rate[i])
    c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]
    y = -s * x_min + c * y_min
    x = np.cross(y, z)
    R_rel = _frame(x, y, z)
    return R_thorax @ R_rel, flex, w


def _leg(hip: np.ndarray, knee_d: np.ndarray, ankle_d: np.ndarray, thigh: float, shank: float, pelvis_y: np.ndarray):
    reach = ankle_d - hip
    dist = np.linalg.norm(reach, axis=-1)
    dist_c = np.clip(dist, abs(thigh - shank) + 1e-3, thigh + shank - 1e-4)
    line = reach / dist[:, None]
    raw = knee_d - hip
    raw = raw - line * np.sum(raw * line, -1, keepdims=True)
    # Near full extension the measured knee plane is noise; hold the last one.
    bend = np.zeros_like(raw)
    prev = pelvis_y[0]
    for i in range(len(raw)):
        n = float(np.linalg.norm(raw[i]))
        w = min(1.0, n / 0.04)
        cand = w * (raw[i] / max(n, 1e-12)) + (1.0 - w) * prev
        cand = cand - line[i] * float(cand @ line[i])
        bend[i] = cand / max(float(np.linalg.norm(cand)), 1e-12)
        prev = bend[i]
    a = (thigh * thigh - shank * shank + dist_c * dist_c) / (2 * dist_c)
    hgt = np.sqrt(np.maximum(thigh * thigh - a * a, 0.0))
    knee = hip + line * a[:, None] + bend * hgt[:, None]
    ankle = hip + line * dist_c[:, None]
    z_t = _unit(hip - knee)
    s = _unit(ankle - knee)
    perp = s - (-z_t) * np.sum(s * -z_t, -1, keepdims=True)
    n = np.linalg.norm(perp, axis=-1, keepdims=True)
    y_t = _unit(bend - z_t * np.sum(bend * z_t, -1, keepdims=True))
    R_thigh = _frame(np.cross(y_t, z_t), y_t, z_t)
    flex = np.arctan2(n[:, 0], np.sum(s * -z_t, -1))
    R_shank = R_thigh @ _rot("x", -flex)
    ys, zs = R_shank[:, :, 1], R_shank[:, :, 2]
    ankle_angle = np.clip(np.arctan2(-ys[:, 2], zs[:, 2]), -0.8, 0.8)
    return R_thigh, flex, ankle_angle


def _hand_angles(R_fore: np.ndarray, pron: np.ndarray, hand_dir: np.ndarray):
    R_rad = R_fore @ _rot("z", pron)
    h = np.einsum("nji,nj->ni", R_rad, hand_dir)
    dev = np.arcsin(np.clip(-h[:, 0], -1.0, 1.0))
    flex = np.arctan2(h[:, 1], -h[:, 2])
    R_hand = R_rad @ _rot("x", flex) @ _rot("y", dev)
    return flex, dev, R_hand


def build_reference(
    points: dict[str, np.ndarray],
    time: np.ndarray,
    events: dict[str, float],
    body: BodyParams,
    subject_height_m: float,
    pronation_deg: np.ndarray | None,
    t0: float,
    t1: float,
    dt: float,
    humeral_rotation_deg: np.ndarray | None = None,
    pronation_offset_deg: float | None = None,
) -> Reference:
    """Joint trajectories for the model from joint centers (model frame, subject's size)."""
    scale = body.height_m / subject_height_m
    pad = 0.03
    keep = (time >= t0 - pad) & (time <= t1 + pad)
    ts = time[keep]
    k_fc = int(np.argmin(np.abs(time)))
    lead0 = points["lead_ankle_jc"][k_fc].copy()
    origin = np.array([points["rear_ankle_jc"][k_fc][0], points["rear_ankle_jc"][k_fc][1], lead0[2]])
    p = {}
    for name, xyz in points.items():
        v = (xyz[keep] - origin) * scale
        v[:, 2] += body.ankle_height_m
        p[name] = v
    pron_series = np.radians(pronation_deg[keep]) if pronation_deg is not None else np.zeros(len(ts))

    def span(a: str, b: str) -> float:
        return float(np.median(np.linalg.norm(p[a] - p[b], axis=1)))

    lengths = {
        "upper_arm": span("shoulder_jc", "elbow_jc"),
        "forearm": span("elbow_jc", "wrist_jc"),
        "glove_upper_arm": span("glove_shoulder_jc", "glove_elbow_jc"),
        "glove_forearm": span("glove_elbow_jc", "glove_wrist_jc"),
        "trunk": float(np.median(np.linalg.norm(p["thorax_prox"] - 0.5 * (p["lead_hip"] + p["rear_hip"]), axis=1))),
        "thigh": 0.5 * (span("lead_hip", "lead_knee_jc") + span("rear_hip", "rear_knee_jc")),
        "shank": 0.5 * (span("lead_knee_jc", "lead_ankle_jc") + span("rear_knee_jc", "rear_ankle_jc")),
        "hip_half": 0.5 * span("lead_hip", "rear_hip"),
    }

    # Trunk.
    R_thorax = _thorax(p)
    R_level = _level_pelvis(p)
    R_rel = np.swapaxes(R_level, -1, -2) @ R_thorax
    axis, swing, twist = _swing_twist(R_rel)
    R_pelvis = R_level @ _axis_angle(axis, PELVIS_SWING_SHARE * swing)
    R_middle = R_level @ _axis_angle(axis, (PELVIS_SWING_SHARE + LUMBAR_SWING_SHARE) * swing) @ _rot("z", LUMBAR_TWIST_SHARE * twist)
    R_upper = R_thorax
    mid_hip = 0.5 * (p["lead_hip"] + p["rear_hip"])
    # Trunk geometry: lumbar joint offset in the pelvis frame and the two
    # upper segment lengths, least squares so the chain ends on thorax_prox.
    # Linear in the unknowns; fixed for the whole window.
    A = np.concatenate([R_pelvis, R_middle[:, :, 2:3], R_upper[:, :, 2:3]], axis=2).reshape(-1, 5)
    sol, *_ = np.linalg.lstsq(A, (p["thorax_prox"] - mid_hip).reshape(-1), rcond=None)
    lumbar_offset, middle_len, upper_len = sol[:3], float(sol[3]), float(sol[4])
    trunk_err = np.linalg.norm((A @ sol).reshape(-1, 3) - (p["thorax_prox"] - mid_hip), axis=1)

    # Girdle: pivot fitted in the thorax frame over the window.
    def girdle(shoulder: str) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        v = np.einsum("nji,nj->ni", R_thorax, p[shoulder] - p["thorax_prox"])
        c, r = _sphere_fit(v)
        if not (0.06 < r < 0.30) or abs(c[0]) > 0.12:
            c, r = np.zeros(3), float(np.mean(np.linalg.norm(v, axis=1)))
        u = v - c
        prot = np.unwrap(np.arctan2(u[:, 1], u[:, 0]))
        elev = -np.arcsin(np.clip(u[:, 2] / np.linalg.norm(u, axis=1), -1, 1))
        return c, r, prot, elev

    c_t, r_t, prot, elev = girdle("shoulder_jc")
    c_g, r_g, gprot, gelev = girdle("glove_shoulder_jc")
    R_clav = R_upper @ _rot("z", prot) @ _rot("y", elev)
    R_gclav = R_upper @ _rot("z", gprot) @ _rot("y", gelev)

    if "elbow_lateral" in p:
        R_hum, elbow, plane_weight = _humerus_blended(
            R_thorax, p["shoulder_jc"], p["elbow_jc"], p["wrist_jc"], None, ts,
            epicondyle_axis=_unit(p["elbow_lateral"] - p["elbow_medial"]),
        )
    elif humeral_rotation_deg is not None:
        er_rate = np.gradient(np.radians(humeral_rotation_deg[keep]), ts)
        R_hum, elbow, plane_weight = _humerus_blended(R_thorax, p["shoulder_jc"], p["elbow_jc"], p["wrist_jc"], er_rate, ts)
    else:
        R_hum, elbow = _limb(p["shoulder_jc"], p["elbow_jc"], p["wrist_jc"])
        plane_weight = np.ones(len(ts))
    R_fore = R_hum @ _rot("x", elbow)
    R_ghum, gelbow = _limb(p["glove_shoulder_jc"], p["glove_elbow_jc"], p["glove_wrist_jc"])
    hand_dir = _unit(p["hand_jc"] - p["wrist_jc"])

    # Offset between OBP's pronation zero and the model's (palm on the
    # flexion side). The per-pitch value that turns the palm most toward home
    # at release is kept as a diagnostic only.
    k_br = int(np.argmin(np.abs(ts - events["BR_time"])))
    best = (-2.0, 0.0)
    for off in np.radians(np.arange(-180.0, 180.0, 1.0)):
        _, _, Rh = _hand_angles(R_fore[k_br : k_br + 1], pron_series[k_br : k_br + 1] + off, hand_dir[k_br : k_br + 1])
        score = float(Rh[0, 1, 1])
        if score > best[0]:
            best = (score, float(off))
    offset = math.radians(PRONATION_ZERO_OFFSET_DEG if pronation_offset_deg is None else pronation_offset_deg)
    pron = pron_series + offset
    _, _, Rh_br = _hand_angles(R_fore[k_br : k_br + 1], pron[k_br : k_br + 1], hand_dir[k_br : k_br + 1])
    wflex, wdev, _ = _hand_angles(R_fore, pron, hand_dir)

    # Legs.
    R_shank = {}
    leg = {}
    for side, sign in (("lead", -1.0), ("trail", 1.0)):
        hip = mid_hip + R_pelvis[:, :, 0] * (sign * lengths["hip_half"])
        key = "lead" if side == "lead" else "rear"
        R_th, knee, ankle = _leg(
            hip, p[f"{key}_knee_jc"], p[f"{key}_ankle_jc"], lengths["thigh"], lengths["shank"], R_pelvis[:, :, 1]
        )
        leg[side] = (R_th, knee, ankle)
        R_shank[side] = R_th @ _rot("x", -knee)

    def rel(Rp: np.ndarray, Rc: np.ndarray) -> np.ndarray:
        return _continuous(quat_from_mat(np.swapaxes(Rp, -1, -2) @ Rc))

    balls_s = {
        "lead_hip": rel(R_pelvis, leg["lead"][0]),
        "trail_hip": rel(R_pelvis, leg["trail"][0]),
        "lumbar": rel(R_pelvis, R_middle),
        "thoracic": rel(R_middle, R_upper),
        "gh": rel(R_clav, R_hum),
        "glove_gh": rel(R_gclav, R_ghum),
    }
    hinges_s = {
        "lead_knee": leg["lead"][1],
        "lead_ankle": leg["lead"][2],
        "trail_knee": leg["trail"][1],
        "trail_ankle": leg["trail"][2],
        "sc_prot": prot,
        "sc_elev": elev,
        "elbow": elbow,
        "pronation": pron,
        "wrist_flex": wflex,
        "wrist_dev": wdev,
        "glove_sc_prot": gprot,
        "glove_sc_elev": gelev,
        "glove_elbow": gelbow,
    }

    grid = t0 + np.arange(int(round((t1 - t0) / dt)) + 1) * dt
    hinge = {}
    for name, series in hinges_s.items():
        sp = Spline(ts, np.unwrap(series))
        hinge[name] = (sp(grid, 0)[:, 0], sp(grid, 1)[:, 0], sp(grid, 2)[:, 0])

    def rotation_track(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qq = _unit(Spline(ts, q)(grid, 0))
        dq = np.gradient(qq, dt, axis=0)
        w = 2.0 * quat_mul(quat_conj(qq), dq)[:, 1:]
        a = np.gradient(w, dt, axis=0)
        return qq, w, a

    ball = {name: rotation_track(q) for name, q in balls_s.items()}
    pos_sp = Spline(ts, mid_hip)
    pq, pw, pa = rotation_track(_continuous(quat_from_mat(R_pelvis)))
    pelvis = {
        "pos": pos_sp(grid, 0),
        "vel": pos_sp(grid, 1),
        "acc": pos_sp(grid, 2),
        "quat": pq,
        "w": pw,
        "a": pa,
    }
    grid_points = {name: Spline(ts, v)(grid, 0) for name, v in p.items()}
    sole = p["rear_ankle_jc"][int(np.argmin(np.abs(ts - t0)))][2] - body.ankle_height_m
    geometry = Geometry(
        clavicle_pivot=c_t,
        clavicle_m=r_t,
        glove_pivot=c_g,
        glove_clavicle_m=r_g,
        rubber_z=max(0.0, float(sole)),
        scale=scale,
        lengths={**lengths, "middle_trunk": middle_len, "upper_trunk": upper_len},
        lumbar_offset=lumbar_offset,
    )
    return Reference(
        t=grid,
        hinge=hinge,
        ball=ball,
        pelvis=pelvis,
        events=dict(events),
        geometry=geometry,
        points=grid_points,
        pronation_offset_deg=math.degrees(offset),
        palm_home_at_release=float(Rh_br[0, 1, 1]),
        extras={
            "plane_weight_at_release": float(plane_weight[k_br]),
            "best_pronation_offset_deg": math.degrees(best[1]),
            "trunk_fit_rms_mm": 1000.0 * float(np.sqrt(np.mean(trunk_err**2))),
        },
    )
