"""Full-body pitching delivery in MuJoCo, driven by measured joint centers.

The skeleton is a right-hander:

    pelvis (free) ─ lumbar (ball) ─ middle trunk ─ thoracic (ball) ─ upper trunk + head
      │                                                 ├─ girdle: sc_prot, sc_elev ─ gh (ball) ─ elbow ─ pronation ─ wrist flex/dev ─ hand
      │                                                 └─ glove girdle ─ glove gh ─ glove elbow ─ glove
      └─ hips (ball) ─ knees ─ ankles

Segment masses, centers of mass, and radii of gyration are de Leva (1996) with
the three-part trunk; the shoulder girdle mass is Veeger (1991). Segment
lengths are the subject's joint-center distances.

Reference motion: published means converted to joint trajectories by
`imitate.build_reference`. Until foot contact the pelvis and the legs are
prescribed, and both soles sit in the mound so the contact force is real.
After contact the pelvis keeps the published rotation, but its translation
is not forced: the feet brake it. The lead knee is a torque-limited hold of
the published flexion. Trunk and shoulder stay on a tracking servo. From
foot contact, pronation is a weak spring plus passive stiffness. The wrist
adds an active flexion and ulnar-deviation torque, timed like DiGiovine's
late-cocking and acceleration EMG and clipped to Delp, on top of the same
passive stiffness. In the finger-roll window the elbow is a capped servo,
and the fingers add a voluntary flexion at their ceiling. The ball's
contact force can push the hand back.
Until the roll the ball is placed in the palm frame, seated on the pads,
and ball contacts are off: a welded ball that the pads penetrate is an
infinite wall, and that force shows up at the shoulder. At the roll,
contacts turn on and the release is the moment contact stops.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

from pitchlab.body import DE_LEVA, GLOVE_KG, BodyParams
from pitchlab.constants import (
    BALL_RADIUS_M,
    CONTACT_MS,
    CONTACT_SOLIMP,
    CONTACT_SOLREF,
    DIP_TORQUE_NM,
    MCP_TORQUE_NM,
    PAD_RADIUS_M,
    PIP_TORQUE_NM,
    SEAM_BUMP_RADIUS_M,
    SEAM_COUNT,
    THUMB_OFF_MS,
    TIMESTEP_S,
)
from pitchlab.balls import contact_mu, get_ball
from pitchlab.fourseam import four_seam_rotation
from pitchlab.grips import GRIPS, Grip
from pitchlab.imitate import BALLS, HINGES, Geometry, Reference, build_reference, quat_conj, quat_from_mat, quat_log, quat_mul
from pitchlab.imitate import _leg as _reach_leg
from pitchlab.motion import Motion
from pitchlab.filters import lowpass
from pitchlab.seams import seam_samples
from pitchlab.simulate import (
    FINGERS,
    JOINT_LIMITS,
    _ball_quat,
    _clock,
    _finger_joint_min,
    _finger_of_pad,
    seam_pair_xml,
    _finger_xml,
    _jid,
    _quat_str,
    _site_pos,
    _stamp,
    _touching,
    tilt_and_efficiency,
    PULP_CENTERS,
    PULP_NODE_RADIUS_M,
)

KINEMATIC_BALLS = ("lead_hip", "trail_hip")
KINEMATIC_HINGES = ("lead_knee", "lead_ankle", "trail_knee", "trail_ankle")
DRIVEN_BALLS = tuple(b for b in BALLS if b not in KINEMATIC_BALLS)
DRIVEN_HINGES = tuple(h for h in HINGES if h not in KINEMATIC_HINGES)
# Servo bandwidth of the PD correction on top of inverse dynamics. Not a
# muscle property; it only keeps the integrated body on the reference.
SERVO_HZ = 40.0
T0_S = -0.06
AFTER_RELEASE_S = 0.10
FLEISIG_CUTOFF_HZ = 13.4  # Fleisig dissertation, kinetics methods
# Hand frame of `simulate._finger_xml` (fingers +X, palm normal +Z, thumb −Y)
# mounted in the wrist body (hand long axis −Z, palm normal +Y, thumb +X).
HAND_MOUNT = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
WRIST_SIZE_M = 0.025
# Middle-finger knuckle along the hand of `simulate.ROOTS`. That hand starts at
# the wrist crease; the joint center is proximal of it, so the hand is mounted
# with the knuckle at de Leva's wrist-to-third-metacarpal length.
KNUCKLE_X_M = 0.066


@dataclass
class DeliveryResult:
    pitch_id: str
    grip: str
    speed_m_s: float
    speed_mph: float
    radar_mph: float
    spin_rpm: float
    spin_efficiency: float
    tilt_deg: float
    clock: str
    velocity: np.ndarray  # model frame (+X throwing side)
    omega: np.ndarray
    separated: bool
    release_ms: float  # ball leaves the fingers, from foot contact
    obp_release_ms: float
    release_height_m: float
    release_side_m: float
    extension_m: float  # ball ahead of the rubber's front edge
    min_finger_rad: float
    peaks: dict[str, tuple[float, float]] = field(default_factory=dict)  # deg/s, ms (integrated)
    reference_peaks: dict[str, tuple[float, float]] = field(default_factory=dict)  # same, on the reference
    mer_deg: float = 0.0
    mer_ms: float = 0.0
    reference_mer_deg: float = 0.0
    separation_at_fc_deg: float = 0.0
    max_separation_deg: float = 0.0
    trunk_forward_at_release_deg: float = 0.0
    trunk_lateral_at_release_deg: float = 0.0
    trunk_rotation_at_release_deg: float = 0.0
    elbow_at_release_deg: float = 0.0
    arm_slot_deg: float = 0.0
    tracking_rms_deg: float = 0.0
    hand_error_mm: float = 0.0  # model hand vs measured hand_jc, rms over the cycle
    loads: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (peak, ms)
    obp_loads: dict[str, tuple[float, float]] = field(default_factory=dict)
    hand_speed_at_release_m_s: float = 0.0
    obp_hand_speed_at_release_m_s: float = 0.0
    contributions: dict[str, np.ndarray] = field(default_factory=dict)
    series: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    height_m: float = 0.0
    mass_kg: float = 0.0
    pronation_offset_deg: float = 0.0
    push_off_ms: float = 0.0  # finger roll ends here: peak speed of the palm-held ball point
    # Same loads low-passed like Fleisig et al. (1995): 13.4 Hz, Winter's dual-pass 2nd order.
    fleisig_loads: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Peak resultant force of each finger pad in the last 15 ms before the ball leaves.
    finger_force_n: dict[str, float] = field(default_factory=dict)
    # Same contacts split the way Kinoshita reports them: resultant near −8 ms,
    # and the normal/shear parts. `raw_max_n` is the unfiltered peak over the
    # whole roll, which includes a terminal collision the papers do not quote.
    finger_force_parts: dict[str, dict[str, float]] = field(default_factory=dict)
    # "npb" (unified Mizuno ball, the default) or "mlb".
    ball_id: str = "npb"
    # Roll-window drive. Empty when distal_forward is off. Otherwise the
    # contact-free inverse-dynamics torque and the clipped torque that was applied.
    distal_drive: dict = field(default_factory=dict)

    @property
    def launch_deg(self) -> float:
        """Vertical launch angle of the ball, degrees (negative is down)."""
        v = self.velocity
        return math.degrees(math.atan2(float(v[2]), math.hypot(float(v[0]), float(v[1]))))

    @property
    def launch_horizontal_deg(self) -> float:
        """Positive is toward the glove side (−X for a right-hander)."""
        v = self.velocity
        return math.degrees(math.atan2(-float(v[0]), float(v[1])))

    def release_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Velocity and spin mirrored into the release-only model's frame (throwing side −X)."""
        v = np.array([-self.velocity[0], self.velocity[1], self.velocity[2]])
        w = np.array([self.omega[0], -self.omega[1], -self.omega[2]])
        return v, w


# ---------------------------------------------------------------- model


# The caps are 80 N × 0.08 m (Kinoshita / Yamaguchi force, assumed moment arm).
# Below a scale of about 0.8 the finger collapses, the ball hits it, and the
# impact force rises further above the measured ~90 N. 0.88 is the lowest
# scale that keeps the launch and the spin axis inside the stage-1 window.
FINGER_CAP_SCALE = 0.88
TORQUE_CAP_NM = {k: v * FINGER_CAP_SCALE for k, v in {"mcp": MCP_TORQUE_NM, "pip": PIP_TORQUE_NM, "dip": DIP_TORQUE_NM}.items()}
# Joint ratios of `simulate._initial_finger`: one flexion parameter per finger.
FLEX_SHARE = {"mcp": 1.0, "pip": 0.85, "dip": 0.55}
# Matsuo et al. 2018: the ball starts rolling up the fingertip 8 ± 1 ms before
# release. 9.0 ms is that mean plus one SD, kept from the pitch that passed
# with the wrist uncapped. With the muscle ceiling applied, that pitch is no
# longer inside the stage-1 window at any lead in 7–9 ms.
ROLL_BEFORE_PUSH_S = 0.0090
# Hold-phase grip servo (same gains as `simulate._actuator`).
HOLD_KP, HOLD_KV = 18.0, 0.35
# Fastball grip: the ball sits on the index and middle pads and the thumb,
# off the palm. Coaching texts call it "a finger's width"; 12 mm is assumed.
PALM_GAP_M = 0.012
PALM_PAD_M = 0.013  # radius of palm_pad in build_model
# Holding fingers servo this far past contact, so they squeeze the ball
# (≈ HOLD_KP × 0.05 = 0.9 N·m at the MCP, ~10 N at the pad; unpublished).
GRIP_SQUEEZE_RAD = 0.05
# Whether ring and pinky extend when the thumb lets go (True) or just stop
# squeezing and stay curled (False). Neither is published; extending swings
# the ring pad into the ball as it rolls forward.
OPEN_EXTENDS_SUPPORT = True
# Rise time of the rolling fingers' torque from the grip to the cap (0 = a step).
ROLL_RAMP_S = 0.0
# From roll start until release these hinges are not position-servoed.
# Positive directions match the model: wrist_flex flexion, wrist_dev ulnar
# (distal axis moves away from the thumb, +X), pronation pronation.
# Off. The minimum acceleration change that puts release torque inside the
# muscle box redirects the palm by about 10 m/s over the 8 ms the ball is
# still on the fingers, and returning to the published curve afterwards takes
# a shoulder acceleration of about 1e4 rad/s². Either way the ball leaves the
# plate. The muscle ceiling is still applied to the published curve.
FEASIBLE_RELEASE = False
YIELD_HINGES = ("elbow", "pronation", "wrist_flex", "wrist_dev")
# Delp, Grierson, Buchanan, J Biomech 29:1371–1375 (1996). The strongest of
# 10 healthy men (the top of the published range). The means were 12.2
# flexion, 7.1 extension, 11.0 radial and 9.5 ulnar. Not pitchers. Pitching
# inverse dynamics is lower: Nissen et al., Am J Sports Med 37:1492 (2009),
# adolescent fastball peaks before MER were flexion 8.3±3.6, ulnar 3.2±1.5,
# pronation 1.7±1.2 N·m. Solomito, Garibay, Woods, Õunpuu, Nissen, Sports
# Biomech 13:320–331 (2014), college fastball ulnar moment 5.1±1.9 N·m
# (curveball 7.3±2.2). Those
# are not used as the clip: an adult release-window moment was not found,
# and the ceiling applied here is Delp's published maximum.
WRIST_FLEX_LIMIT_NM = 18.7
WRIST_EXT_LIMIT_NM = 9.4
WRIST_RADIAL_LIMIT_NM = 15.3
WRIST_ULNAR_LIMIT_NM = 11.9
# Fleisig dissertation Table 2: peak elbow flexion torque during arm
# acceleration 61 ± 11 N·m. Holzbaur, Murray, Delp, Ann Biomed Eng 33:829
# (2005) cite an experimental isometric extension peak of 52 N·m and a
# flexion peak near 77 N·m. Positive elbow torque is flexion.
# Holzbaur, Murray, Delp (Ann Biomed Eng 33:829, 2005) cite an experimental
# flexion peak near 77 N·m. Their model's own flexion peak is 103 N·m and is
# not used. Extension stays the experimental 52 N·m they cite.
ELBOW_FLEX_LIMIT_NM = 77.0
ELBOW_EXT_LIMIT_NM = 52.0
# Holzbaur 2005: supination moment-generating capacity is 8 N·m with the arm
# in neutral rotation at 0° of elbow flexion. Pronation's peak is only in
# their Figure 5B, so it is capped at that same stated magnitude.
PRONATION_LIMIT_NM = 8.0
SUPINATION_LIMIT_NM = 8.0
# Formica, Charles, Zollo, Guglielmelli, Hogan, J Neurophysiol 108:1158 (2012).
# Linearized passive stiffness about neutral: FE 0.96, RUD 1.49 N·m/rad
# (the values Drake's thesis tabulates from that paper). Pronation-supination
# stiffness is similar to FE and below RUD (Drake and Charles), so PS uses
# the FE number. Charles and Hogan (J Biomech 44:614, 2011) find stiffness
# dominates small posture adjustments. That damper does not bound the
# velocity product of a pitching shoulder, so the hinge damping below is a
# separate guess: 1 N·m·s/rad, about the size that levels a 50 N·m product
# near 50 rad/s. It is applied as MuJoCo joint damping so the implicit
# integrator sees it.
PASSIVE_K_NM_PER_RAD = {"wrist_flex": 0.96, "wrist_dev": 1.49, "pronation": 0.96}
PASSIVE_B_NM_S_PER_RAD = 0.0
WHIP_DAMPING = 1.0
# Weak spring toward the published angle. One radian of lag is 4 N·m, inside
# Delp's box, so a velocity product larger than the cap moves the joint.
WHIP_ACT_K_NM_PER_RAD = 4.0
WHIP_HINGES = ("pronation", "wrist_flex", "wrist_dev")
# DiGiovine, Jobe, Pink, Perry, J Shoulder Elbow Surg 1:15–25 (1992).
# Their bins are low 0–20%, moderate 21–40%, high 41–60%, very high >60% of
# a manual muscle test. FCR and FCU sit in the high band in late cocking
# (foot contact to maximum external rotation) and stay high in acceleration
# (MER to release). The paper's phase means are not in the abstract, so the
# peak fraction of Delp's isometric maximum is a motor command inside that
# cap, not a copied cell. Onset is foot contact. The ramp reaches the peak
# at MER and holds through release. A 40 ms rise at the end of early cocking
# is a guess so the command is not a step.
WRIST_EMG_PEAK = {"wrist_flex": 1.0, "wrist_dev": 0.15}
WRIST_EMG_RAMP_S = 0.040
# Ligament end-range, AAOS typical values, so a passive wrist cannot spin past
# the joint. Flexion 80°, extension 70°, radial 20°, ulnar 30°, pronation and
# supination 80°. Positive model signs: flexion, ulnar deviation, pronation.
WRIST_ROM_RAD = {
    "wrist_flex": (-math.radians(70.0), math.radians(80.0)),
    "wrist_dev": (-math.radians(20.0), math.radians(30.0)),
    "pronation": (-math.radians(80.0), math.radians(80.0)),
}
# Round isometric bounds, not a pitching-study mean. The lead knee is a
# position hold toward the published flexion, clipped here. The reference
# inverse dynamics wants 2–4 kN·m; that torque is not applied.
KNEE_EXT_LIMIT_NM = 250.0
KNEE_FLEX_LIMIT_NM = 200.0
HIP_LIMIT_NM = 400.0
# 300 N·m let the ground moment yaw the trunk. 600 N·m is still a guess,
# above one hip, shared by both hips and the trunk.
PELVIS_ANG_LIMIT_NM = 600.0
ANKLE_LIMIT_NM = 160.0
LEG_SERVO_HZ = 12.0
# The published ankle sits on the sole. Commanding the lead foot this far
# through the mound is what makes the contact carry weight: the ground stops
# the foot, and the force is the GRF. 8 mm is a guess, checked against the
# measured vertical load. The pivot foot is not pressed this way.
PLANT_PRESS_M = 0.008
# "braced": after foot contact the pelvis translation is left to foot–mound
# contact and the legs are torque-capped servos (iterations 15–18).
# "guided": pelvis and legs follow the published reference for the whole
# delivery, and trunk, girdle and arm track it with computed torque at
# SERVO_HZ. Foot–mound contact is off; the ground force is the root wrench of
# the inverse dynamics, shared between the feet by the lever rule on the COM.
BODY_MODES = ("braced", "guided")


def _finger_angle(model: mujoco.MjModel, data: mujoco.MjData, finger: str, flex: float) -> None:
    for joint, share in FLEX_SHARE.items():
        lo, hi = JOINT_LIMITS[joint]
        data.qpos[int(model.jnt_qposadr[_jid(model, f"{finger}_{joint}")])] = float(np.clip(share * flex, lo, hi))


def _fingertip_grip(model: mujoco.MjModel, data: mujoco.MjData, grip: Grip) -> dict[str, float]:
    """Flex the rolling fingers and the thumb until the ball touches their pads
    and clears the palm by PALM_GAP_M; place it there. Returns the flexion per finger.

    The other fingers keep the grip's curl (they leave with the thumb).
    """
    hand = _body(model, "hand")
    sids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{n}_pad") for n in FINGERS}
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "palm")
    reach = BALL_RADIUS_M + PAD_RADIUS_M
    holders = (*grip.roll_fingers, "thumb")
    flex = {f: math.radians(grip.mcp_deg[f]) for f in FINGERS}
    best = None
    # Fingertips on the ball, a finger's width off the palm. The horseshoe is
    # oriented after this, from the finger directions. A search that also
    # forced the second horseshoe leg onto a proximal point moved the spin
    # axis off 1:00 and is not used: on this hand that leg does not lie on
    # both pads at once (see fourseam.py).
    for s in np.radians(np.arange(0.0, 90.1, 2.5)):
        for s_t in np.radians(np.arange(-10.0, 90.1, 2.5)):
            for f in grip.roll_fingers:
                _finger_angle(model, data, f, s)
            _finger_angle(model, data, "thumb", s_t)
            mujoco.mj_kinematics(model, data)
            pads = [data.site_xpos[sids[f]].copy() for f in holders]
            normal = np.array(data.site_xmat[palm]).reshape(3, 3)[:, 2]
            c = np.mean(pads, 0) + 0.03 * normal
            anchors = [*pads, data.site_xpos[palm].copy()]
            radii = [reach] * len(pads) + [BALL_RADIUS_M + PALM_PAD_M + PALM_GAP_M]
            for _ in range(30):
                d = [c - a for a in anchors]
                res = np.array([np.linalg.norm(x) - r for x, r in zip(d, radii)])
                jac = np.array([x / np.linalg.norm(x) for x in d])
                c = c - np.linalg.lstsq(jac, res, rcond=None)[0]
            res = np.array([np.linalg.norm(c - a) - r for a, r in zip(anchors, radii)])
            err = float(np.sum(res * res))
            if float((c - data.site_xpos[palm]) @ normal) > 0 and (best is None or err < best[0]):
                best = (err, s, s_t, c)
    _, s, s_t, center = best
    for f in grip.roll_fingers:
        flex[f] = s
    flex["thumb"] = s_t
    for f in FINGERS:
        _finger_angle(model, data, f, flex[f])
    # The other fingers curl until they touch the side of the ball.
    for f in FINGERS:
        if f in holders:
            continue
        for s_f in np.radians(np.arange(grip.mcp_deg[f], 120.1, 2.5)):
            _finger_angle(model, data, f, s_f)
            mujoco.mj_kinematics(model, data)
            if np.linalg.norm(data.site_xpos[sids[f]] - center) < reach:
                break
            flex[f] = s_f
        _finger_angle(model, data, f, flex[f])
    mujoco.mj_forward(model, data)
    R_hand = np.array(data.xmat[hand]).reshape(3, 3)
    adr = int(model.jnt_qposadr[_jid(model, "ball_free")])
    data.qpos[adr : adr + 3] = center
    if grip.id == "four-seam":
        # The euler on the grip is not the four-seam pose. The seams are placed
        # from the finger directions (see fourseam.py).
        dirs = {}
        axes = {}
        for name in ("index", "middle", "thumb"):
            sid = sids[name]
            dirs[name] = R_hand.T @ (data.site_xpos[sid] - center)
            axes[name] = R_hand.T @ np.array(data.site_xmat[sid]).reshape(3, 3)[:, 0]
        R_ball = four_seam_rotation(
            dirs["index"], dirs["middle"], axes["index"], axes["middle"], dirs["thumb"],
        )
    else:
        R_ball = np.zeros(9)
        mujoco.mju_quat2Mat(R_ball, _ball_quat(grip))
        R_ball = R_ball.reshape(3, 3)
    data.qpos[adr + 3 : adr + 7] = quat_from_mat(R_hand @ R_ball)
    mujoco.mj_forward(model, data)
    _seat_pulp(model, data)
    return flex


def _seat_pulp(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Pull proud pulp nodes back to the ball surface before the roll.

    The nodes stand 1 mm outside the backing the grip aims at, so a seam under
    a pad begins inside a node. That overlap is an impact on the first free
    step, not the squeeze Kinoshita measured. Retract each penetrating node
    along its slide until the contact opens. The spring still restores at 4 N.
    """
    for _ in range(12):
        mujoco.mj_forward(model, data)
        moved = False
        for k in range(data.ncon):
            con = data.contact[k]
            if float(con.dist) >= -1e-6:
                continue
            for gid in (int(con.geom1), int(con.geom2)):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if "_pad_" not in name:
                    continue
                finger, _, idx = name.partition("_pad_")
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{finger}_pulp_{idx}")
                if jid < 0:
                    continue
                adr = int(model.jnt_qposadr[jid])
                lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
                q = float(np.clip(data.qpos[adr] + float(con.dist), lo, hi))
                if q < float(data.qpos[adr]) - 1e-7:
                    data.qpos[adr] = q
                    moved = True
        if not moved:
            break
    mujoco.mj_forward(model, data)


def _lift_off_pads(
    data: mujoco.MjData,
    pos: np.ndarray,
    pad_ids: list[tuple[int, float]],
    belly_ids: list[int],
    allow_m: float,
) -> np.ndarray:
    """Move the ball center out of the pads, leaving `allow_m` of overlap.

    Holding the ball on the palm pose with no give makes it an infinite wall:
    the arm tracking drives the finger bones through it. The center stays on
    that pose until a pad would sink past `allow_m`. Zero sits the leather on
    the pad. Contacts stay off while this placement runs.
    """
    pos = np.array(pos, float, copy=True)
    for _ in range(6):
        moved = False
        for sid, radius in pad_ids:
            point = data.site_xpos[sid]
            delta = pos - point
            dist = float(np.linalg.norm(delta))
            limit = radius - allow_m
            if dist < limit:
                pos = point + delta / max(dist, 1e-9) * limit
                moved = True
        for sid in belly_ids:
            rf = np.array(data.site_xmat[sid]).reshape(3, 3)
            origin = data.site_xpos[sid]
            for local in PULP_CENTERS:
                point = origin + rf @ local
                delta = pos - point
                dist = float(np.linalg.norm(delta))
                limit = BALL_RADIUS_M + PULP_NODE_RADIUS_M - allow_m
                if dist < limit:
                    pos = point + delta / max(dist, 1e-9) * limit
                    moved = True
            # Leather capsule under the same patch (four-seam only; harmless if absent).
            point = origin + rf @ np.array([-0.008, -0.002, 0.006])
            delta = pos - point
            dist = float(np.linalg.norm(delta))
            limit = BALL_RADIUS_M + 0.0045 - allow_m
            if dist < limit:
                pos = point + delta / max(dist, 1e-9) * limit
                moved = True
        if not moved:
            break
    return pos


def _turn_ball(model: mujoco.MjModel, data: mujoco.MjData, yaw_deg: float, tilt_deg: float) -> None:
    """Rotate the gripped ball about the palm normal, then about the finger line."""
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "palm")
    normal = np.array(data.site_xmat[palm]).reshape(3, 3)[:, 2].copy()
    normal /= np.linalg.norm(normal)
    index = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "index_pad")]
    middle = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "middle_pad")]
    across = middle - index
    across = across - normal * float(across @ normal)
    across /= max(float(np.linalg.norm(across)), 1e-9)
    adr = int(model.jnt_qposadr[_jid(model, "ball_free")])
    base = data.qpos[adr + 3 : adr + 7].copy()
    qy, qt, q, q2 = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(qy, normal, math.radians(yaw_deg))
    mujoco.mju_axisAngle2Quat(qt, across, math.radians(tilt_deg))
    mujoco.mju_mulQuat(q, qy, base)
    mujoco.mju_mulQuat(q2, qt, q)
    data.qpos[adr + 3 : adr + 7] = q2


def _segment(name: str, mass: float, length: float, com_from_joint: float, axis_sign: float = 1.0) -> str:
    """Inertial along the body z axis. de Leva radii are fractions of segment length."""
    s = DE_LEVA[name]
    ix = mass * (s.r_trans * length) ** 2
    iy = mass * (s.r_sag * length) ** 2
    iz = mass * (s.r_long * length) ** 2
    return f'<inertial pos="0 0 {axis_sign * com_from_joint:.4f}" mass="{mass:.4f}" diaginertia="{ix:.6f} {iy:.6f} {iz:.6f}"/>'


def _f(v) -> str:
    return " ".join(f"{float(x):.5f}" for x in v)


def _leg(side: str, hip_x: float, thigh: float, shank: float, body: BodyParams) -> str:
    foot = body.length("foot")
    rgba = "0.62 0.64 0.70 1" if side == "lead" else "0.52 0.54 0.62 1"
    fm = body.mass("foot")
    return f"""
      <body name="{side}_thigh" pos="{hip_x:.4f} 0 0">
        <joint name="{side}_hip" type="ball" damping="0"/>
        {_segment("thigh", body.mass("thigh"), thigh, DE_LEVA["thigh"].com_frac * thigh, -1.0)}
        <geom type="capsule" fromto="0 0 0 0 0 {-thigh:.4f}" size="0.060" density="0" rgba="{rgba}" contype="0" conaffinity="0"/>
        <body name="{side}_shank" pos="0 0 {-thigh:.4f}">
          <joint name="{side}_knee" type="hinge" axis="-1 0 0" limited="true" range="15 178" solreflimit="0.004 1" solimplimit="0.95 0.99 0.02" damping="2"/>
          {_segment("shank", body.mass("shank"), shank, DE_LEVA["shank"].com_frac * shank, -1.0)}
          <geom type="capsule" fromto="0 0 0 0 0 {-shank:.4f}" size="0.045" density="0" rgba="{rgba}" contype="0" conaffinity="0"/>
          <site name="{side}_knee" pos="0 0 0" size="0.01"/>
          <body name="{side}_foot" pos="0 0 {-shank:.4f}">
            <joint name="{side}_ankle" type="hinge" axis="1 0 0" damping="0"/>
            <inertial pos="0 {0.35 * foot:.4f} -0.03" mass="{fm:.3f}" diaginertia="{fm * (0.245 * foot) ** 2:.6f} {fm * 0.0012:.6f} {fm * (0.257 * foot) ** 2:.6f}"/>
            <geom name="{side}_shoe" type="box" pos="0 {0.30 * foot:.4f} {-body.ankle_height_m + 0.022:.4f}" size="0.048 {0.52 * foot:.4f} 0.022"
                  density="0" rgba="0.18 0.17 0.17 1" contype="0" conaffinity="0"/>
            <geom name="{side}_foot" type="sphere" pos="0 0 0" size="{body.ankle_height_m + 0.001:.4f}"
                  density="0" rgba="0.18 0.17 0.17 0" friction="1.1 0.005 0.0001" condim="4"
                  solref="0.03 1" solimp="0.90 0.98 0.010" contype="4" conaffinity="8" margin="0"/>
            <site name="{side}_ankle" pos="0 0 0" size="0.01"/>
          </body>
        </body>
      </body>"""


def build_model(grip: Grip, body: BodyParams, geom: Geometry, friction: float | None = None, pulp: bool = True, ball=None) -> mujoco.MjModel:
    L = geom.lengths
    Lu, Lm = L["upper_trunk"], L["middle_trunk"]
    lumbar = geom.lumbar_offset
    Ll = float(np.linalg.norm(lumbar))
    Lh = body.length("head")
    ua, fa = L["upper_arm"], L["forearm"]
    gua, gfa = L["glove_upper_arm"], L["glove_forearm"]
    hand_len = body.length("hand")
    girdle = body.girdle_mass
    upper_m = body.mass("upper_trunk") - 2.0 * girdle
    finger_mass = 5 * (0.012 + 0.008 + 0.005 + 0.004) + 0.004
    hand_m = max(0.05, body.mass("hand") - finger_mass)
    mount = _quat_str(quat_from_mat(HAND_MOUNT))

    spec, friction, seam_mu = contact_mu(ball, friction)
    seam_center = spec.radius_m - (SEAM_BUMP_RADIUS_M - spec.seam_height_m)
    seams = []
    for i, point in enumerate(seam_samples(SEAM_COUNT)):
        pos = point * seam_center
        seams.append(f'<geom class="seam" name="seam_{i}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}"/>')
    fingers = "\n".join(
        _finger_xml(name, grip.abduction_deg[name], pulp, belly_leather=grip.id == "four-seam") for name in FINGERS
    )
    acts = []
    for name in FINGERS:
        for joint, cap in TORQUE_CAP_NM.items():
            lo, hi = JOINT_LIMITS[joint]
            acts.append(
                f'<position name="{name}_{joint}_act" joint="{name}_{joint}" kp="{HOLD_KP}" kv="{HOLD_KV}" '
                f'forcerange="-{cap} {cap}" ctrlrange="{lo:.4f} {hi:.4f}"/>'
            )
    fr = f"{friction:.4f} 0.02 0.001"
    pairs = seam_pair_xml(seam_mu, pulp)
    c, r = geom.clavicle_pivot, geom.clavicle_m
    gc, gr = geom.glove_pivot, geom.glove_clavicle_m
    head_com = (1.0 - DE_LEVA["head"].com_frac) * Lh
    hm = body.mass("head")
    rz = geom.rubber_z
    skin = 'density="0" contype="0" conaffinity="0" rgba="0.93 0.77 0.65 1"'
    jersey = 'density="0" contype="0" conaffinity="0" rgba="0.16 0.25 0.45 1"'
    xml = f"""
<mujoco model="pitch-delivery">
  <option timestep="{TIMESTEP_S}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" iterations="50"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.55 0.55 0.52" diffuse="0.70 0.68 0.62"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.45 0.62 0.82" rgb2="0.93 0.90 0.82" width="512" height="256"/>
  </asset>
  <default>
    <geom rgba="0.86 0.55 0.42 1"/>
    <default class="pad">
      <geom type="sphere" friction="{fr}" condim="4" solref="{CONTACT_SOLREF}" solimp="{CONTACT_SOLIMP}"
            contype="2" conaffinity="1" rgba="0.90 0.48 0.36 1" mass="0.004"/>
    </default>
    <default class="seam">
      <geom type="sphere" size="{SEAM_BUMP_RADIUS_M}" friction="{seam_mu:.4f} 0.02 0.001" condim="4"
            solref="{CONTACT_SOLREF}" solimp="{CONTACT_SOLIMP}"
            contype="0" conaffinity="0" rgba="0.72 0.14 0.12 1" mass="1e-6"/>
    </default>
  </default>
  <worldbody>
    <light pos="2.0 -1.0 4.5" dir="-0.35 0.3 -1"/>
    <geom name="ground" type="plane" size="8 14 0.1" pos="0 6 0" rgba="0.36 0.48 0.30 1" contype="0" conaffinity="0"/>
    <geom name="mound" type="box" size="1.8 1.7 0.050" pos="0 0.15 {rz - 0.050:.4f}" rgba="0.58 0.44 0.31 1" friction="1.1 0.005 0.0001" condim="4" solref="0.03 1" solimp="0.90 0.98 0.010" contype="8" conaffinity="4" margin="0"/>
    <geom name="slope" type="box" size="1.4 {max(rz, 0.01) * 6:.4f} 0.01" pos="0 {1.1 + max(rz, 0.01) * 6:.4f} {rz / 2:.4f}" euler="{-math.degrees(math.atan(1 / 12)):.3f} 0 0" rgba="0.58 0.44 0.31 1" contype="0" conaffinity="0"/>
    <geom name="rubber" type="box" size="0.30 0.08 0.012" pos="0 0 {rz + 0.006:.4f}" rgba="0.95 0.95 0.93 1" contype="0" conaffinity="0"/>
    <geom name="lane" type="box" size="0.04 6.5 0.004" pos="0 8.0 0.006" rgba="0.82 0.78 0.62 1" contype="0" conaffinity="0"/>
    <body name="ball">
      <freejoint name="ball_free"/>
      <geom name="leather" type="sphere" size="{spec.radius_m}" mass="{spec.mass_kg}"
            friction="{fr}" condim="4" solref="{CONTACT_SOLREF}" solimp="{CONTACT_SOLIMP}"
            contype="1" conaffinity="2" rgba="0.95 0.93 0.88 1"/>
      <site name="ball_center" pos="0 0 0"/>
      {''.join(seams)}
    </body>
    <body name="pelvis">
      <freejoint name="pelvis_free"/>
      {_segment("lower_trunk", body.mass("lower_trunk"), Ll, (1 - DE_LEVA["lower_trunk"].com_frac) * Ll)}
      <geom type="capsule" fromto="{-L["hip_half"]:.4f} 0 0 {L["hip_half"]:.4f} 0 0" size="0.075" density="0" rgba="0.85 0.85 0.82 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0.02 {_f(lumbar)}" size="0.12" density="0" rgba="0.85 0.85 0.82 1" contype="0" conaffinity="0"/>
      <site name="pelvis" pos="0 0 0" size="0.01"/>
      {_leg("lead", -L["hip_half"], L["thigh"], L["shank"], body)}
      {_leg("trail", L["hip_half"], L["thigh"], L["shank"], body)}
      <body name="middle_trunk" pos="{_f(lumbar)}">
        <joint name="lumbar" type="ball" damping="0"/>
        {_segment("middle_trunk", body.mass("middle_trunk"), Lm, (1 - DE_LEVA["middle_trunk"].com_frac) * Lm)}
        <geom type="capsule" fromto="0 0 0 0 0 {Lm:.4f}" size="0.125" {jersey}/>
        <body name="upper_trunk" pos="0 0 {Lm:.4f}">
          <joint name="thoracic" type="ball" damping="0"/>
          {_segment("upper_trunk", upper_m, Lu, (1 - DE_LEVA["upper_trunk"].com_frac) * Lu)}
          <geom type="capsule" fromto="0 0 0 0 0 {Lu - 0.02:.4f}" size="0.13" {jersey}/>
          <site name="thorax_prox" pos="0 0 {Lu:.4f}" size="0.01"/>
          <body name="head" pos="0 0 {Lu:.4f}">
            <inertial pos="0 0 {head_com:.4f}" mass="{hm:.3f}" diaginertia="{hm * (0.376 * Lh) ** 2:.6f} {hm * (0.362 * Lh) ** 2:.6f} {hm * (0.312 * Lh) ** 2:.6f}"/>
            <geom type="capsule" fromto="0 0 0 0 0.01 0.09" size="0.045" {skin}/>
            <geom type="sphere" pos="0 0.02 {head_com + 0.02:.4f}" size="0.10" {skin}/>
            <geom type="box" pos="0 0.08 {head_com + 0.06:.4f}" size="0.09 0.06 0.012" density="0" rgba="0.12 0.16 0.30 1" contype="0" conaffinity="0"/>
            <geom type="sphere" pos="0 0.0 {head_com + 0.07:.4f}" size="0.095" density="0" rgba="0.12 0.16 0.30 1" contype="0" conaffinity="0"/>
            <site name="head" pos="0 0.02 {head_com:.4f}" size="0.01"/>
          </body>
          <body name="clavicle" pos="{_f(np.array([0, 0, Lu]) + c)}">
            <joint name="sc_prot" type="hinge" axis="0 0 1" damping="0"/>
            <joint name="sc_elev" type="hinge" axis="0 1 0" damping="0"/>
            <inertial pos="{r * 0.6:.4f} 0 0" mass="{girdle:.3f}" diaginertia="{girdle * 0.002:.6f} {girdle * (0.3 * r) ** 2 + girdle * 0.002:.6f} {girdle * (0.3 * r) ** 2 + girdle * 0.002:.6f}"/>
            <geom type="capsule" fromto="0 0 0 {r:.4f} 0 0" size="0.045" {jersey}/>
            <body name="humerus" pos="{r:.4f} 0 0">
              <joint name="gh" type="ball" damping="0"/>
              <site name="shoulder" pos="0 0 0" size="0.012"/>
              {_segment("upper_arm", body.mass("upper_arm"), ua, DE_LEVA["upper_arm"].com_frac * ua, -1.0)}
              <geom type="capsule" fromto="0 0 0 0 0 {-ua:.4f}" size="0.042" {jersey}/>
              <body name="forearm" pos="0 0 {-ua:.4f}">
                <joint name="elbow" type="hinge" axis="1 0 0" damping="0"/>
                <site name="elbow" pos="0 0 0" size="0.01"/>
                {_segment("forearm", body.mass("forearm"), fa, DE_LEVA["forearm"].com_frac * fa, -1.0)}
                <geom type="capsule" fromto="0 0 0 0 0 {-fa:.4f}" size="0.030" {skin}/>
                <body name="radius" pos="0 0 {-fa:.4f}">
                  <joint name="pronation" type="hinge" axis="0 0 1" damping="{WHIP_DAMPING}"/>
                  <inertial pos="0 0 0" mass="0.03" diaginertia="2e-5 2e-5 2e-5"/>
                  <body name="wrist">
                    <joint name="wrist_flex" type="hinge" axis="1 0 0" damping="{WHIP_DAMPING}" limited="true" range="-70 80" solreflimit="0.01 1" solimplimit="0.9 0.99 0.05"/>
                    <joint name="wrist_dev" type="hinge" axis="0 1 0" damping="{WHIP_DAMPING}" limited="true" range="-20 30" solreflimit="0.01 1" solimplimit="0.9 0.99 0.05"/>
                    {_segment("hand", hand_m, hand_len, 0.5 * hand_len, -1.0)}
                    <site name="wrist" pos="0 0 0" size="0.008"/>
                    <geom type="sphere" size="{WRIST_SIZE_M}" {skin}/>
                    <body name="hand" pos="0 0 {-(hand_len - KNUCKLE_X_M):.4f}" quat="{mount}">
                      <geom type="box" pos="0.034 0 0.004" size="0.038 0.034 0.010" {skin}/>
                      <geom class="pad" name="palm_pad" pos="0.030 0.000 0.016" size="0.013"/>
                      <site name="palm" pos="0.030 0.000 0.016" size="0.006"/>
                      {fingers}
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
          <body name="glove_clavicle" pos="{_f(np.array([0, 0, Lu]) + gc)}">
            <joint name="glove_sc_prot" type="hinge" axis="0 0 1" damping="0"/>
            <joint name="glove_sc_elev" type="hinge" axis="0 1 0" damping="0"/>
            <inertial pos="{gr * 0.6:.4f} 0 0" mass="{girdle:.3f}" diaginertia="{girdle * 0.002:.6f} {girdle * (0.3 * gr) ** 2 + girdle * 0.002:.6f} {girdle * (0.3 * gr) ** 2 + girdle * 0.002:.6f}"/>
            <geom type="capsule" fromto="0 0 0 {gr:.4f} 0 0" size="0.045" {jersey}/>
            <body name="glove_humerus" pos="{gr:.4f} 0 0">
              <joint name="glove_gh" type="ball" damping="0"/>
              <site name="glove_shoulder" pos="0 0 0" size="0.012"/>
              {_segment("upper_arm", body.mass("upper_arm"), gua, DE_LEVA["upper_arm"].com_frac * gua, -1.0)}
              <geom type="capsule" fromto="0 0 0 0 0 {-gua:.4f}" size="0.042" {jersey}/>
              <body name="glove_forearm" pos="0 0 {-gua:.4f}">
                <joint name="glove_elbow" type="hinge" axis="1 0 0" damping="0"/>
                {_segment("forearm", body.mass("forearm") + body.mass("hand") + GLOVE_KG, gfa, 0.75 * gfa, -1.0)}
                <geom type="capsule" fromto="0 0 0 0 0 {-gfa:.4f}" size="0.030" {skin}/>
                <geom type="ellipsoid" pos="0 0 {-gfa - 0.09:.4f}" size="0.075 0.04 0.12" density="0" rgba="0.48 0.30 0.16 1" contype="0" conaffinity="0"/>
                <site name="glove" pos="0 0 {-gfa - 0.09:.4f}" size="0.01"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <contact>
    {pairs}
  </contact>
  <actuator>
    {''.join(acts)}
  </actuator>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)


# ---------------------------------------------------------------- helpers


class _Joints:
    """qpos/qvel addresses of the skeleton joints."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.q = {}
        self.v = {}
        for name in (*HINGES, *BALLS, "pelvis_free", "ball_free"):
            jid = _jid(model, name)
            self.q[name] = int(model.jnt_qposadr[jid])
            self.v[name] = int(model.jnt_dofadr[jid])
        width = {"pelvis_free": 6, "ball_free": 6, **{b: 3 for b in BALLS}, **{h: 1 for h in HINGES}}
        self.skeleton = np.array(
            [self.v[n] + i for n in ("pelvis_free", *BALLS, *HINGES) for i in range(width[n])], dtype=int
        )


def _reach_plant(
    side: str,
    sign: float,
    hip_half: float,
    thigh: float,
    shank: float,
    pelvis_pos: np.ndarray,
    R_pel: np.ndarray,
    ankle_w: np.ndarray,
    knee_w: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Hip quaternion and hinge angles that put this ankle on `ankle_w`.

    The published joint angles assume the published pelvis. Once translation
    is free, those angles lift the sole. Reaching from the pelvis where it
    actually is keeps the foot on the mound, and the contact force is what
    brakes the body.
    """
    hip_w = pelvis_pos + R_pel @ np.array([sign * hip_half, 0.0, 0.0])
    R_th, flex, ankle = _reach_leg(
        hip_w[None, :],
        knee_w[None, :],
        ankle_w[None, :],
        thigh,
        shank,
        R_pel[:, 1][None, :],
    )
    q_hip = quat_from_mat(R_pel.T @ R_th[0])
    return q_hip, float(flex[0]), float(ankle[0])


def _set_reference_state(data: mujoco.MjData, J: _Joints, ref: Reference, i: int, names_ball, names_hinge, pelvis: bool) -> None:
    if pelvis:
        a, b = J.q["pelvis_free"], J.v["pelvis_free"]
        data.qpos[a : a + 3] = ref.pelvis["pos"][i]
        data.qpos[a + 3 : a + 7] = ref.pelvis["quat"][i]
        data.qvel[b : b + 3] = ref.pelvis["vel"][i]
        data.qvel[b + 3 : b + 6] = ref.pelvis["w"][i]
    for name in names_ball:
        q, w, _ = ref.ball[name]
        data.qpos[J.q[name] : J.q[name] + 4] = q[i]
        data.qvel[J.v[name] : J.v[name] + 3] = w[i]
    for name in names_hinge:
        q, qd, _ = ref.hinge[name]
        data.qpos[J.q[name]] = q[i]
        data.qvel[J.v[name]] = qd[i]


def _body(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _rot(data: mujoco.MjData, bid: int) -> np.ndarray:
    return np.array(data.xmat[bid]).reshape(3, 3)


def _body_omega(model: mujoco.MjModel, data: mujoco.MjData, bid: int) -> np.ndarray:
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, res, 0)
    return res[:3].copy()


def _site_velocity(model: mujoco.MjModel, data: mujoco.MjData, sid: int) -> tuple[np.ndarray, np.ndarray]:
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, sid, res, 0)
    return res[:3].copy(), res[3:].copy()


def _interaction(model: mujoco.MjModel, data: mujoco.MjData, bid: int, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Force and moment (about `point`) that the parent applies to body `bid`, world frame."""
    c = data.subtree_com[int(model.body_rootid[bid])]
    moment = np.array(data.cfrc_int[bid, :3])
    force = np.array(data.cfrc_int[bid, 3:])
    return force, moment + np.cross(c - point, force)


def fleisig_er_deg(anterior: np.ndarray, superior: np.ndarray, shoulder, elbow, wrist) -> float:
    """External rotation as Fleisig defines it.

    Angle between the trunk's anterior direction and the forearm, both
    projected onto the plane perpendicular to the upper arm. 90° is the
    forearm along the trunk's superior direction.
    """
    h = elbow - shoulder
    h = h / np.linalg.norm(h)

    def proj(v: np.ndarray) -> np.ndarray:
        v = v - h * float(v @ h)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    f = proj(wrist - elbow)
    # Range (−90°, 270°]: early in the stride the arm sits slightly below 0°.
    return (math.degrees(math.atan2(float(f @ proj(superior)), float(f @ proj(anterior)))) + 90.0) % 360.0 - 90.0


def _yaw(forward: np.ndarray) -> float:
    """Heading of a forward vector; 0 is toward home, positive is counter-clockwise from above."""
    return math.atan2(-float(forward[0]), float(forward[1]))


def _thorax_from_points(p: dict[str, np.ndarray], i: int) -> np.ndarray:
    z = p["thorax_prox"][i] - p["thorax_dist"][i]
    z /= np.linalg.norm(z)
    x = p["thorax_ap"][i] - p["thorax_prox"][i]
    x -= z * float(x @ z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


# ---------------------------------------------------------------- simulation


def _reference(motion: Motion, body: BodyParams, offset_deg: float | None, t1: float, t0: float | None = None) -> Reference:
    return build_reference(
        motion.points,
        motion.time,
        motion.events,
        body,
        motion.height_m,
        motion.pronation_deg,
        t0=T0_S if t0 is None else t0,
        t1=t1,
        dt=TIMESTEP_S,
        humeral_rotation_deg=motion.humeral_rotation_deg,
        pronation_offset_deg=offset_deg,
    )


def prepare(motion: Motion, body: BodyParams | None = None, grip: Grip | str = "four-seam") -> tuple[BodyParams, Reference]:
    body = body or BodyParams(height_m=motion.height_m, mass_kg=motion.mass_kg)
    offset = motion.pronation_offset_deg
    if motion.pronation_deg is None and offset is None:
        offset = solve_pronation(motion, grip)
        motion.pronation_offset_deg = float(offset)
    # The whole clip, including the stride. Pronation search stays on T0_S.
    return body, _reference(motion, body, offset, motion.release + AFTER_RELEASE_S, float(motion.time[0]))


def _ball_point_track(model, data, J: _Joints, ref: Reference, local: np.ndarray, palm_sid: int, lo: float, hi: float, stride: int = 4):
    """Path of a ball held rigidly in the palm while the body follows the reference."""
    idx = np.where((ref.t >= lo) & (ref.t <= hi))[0][::stride]
    X = np.zeros((len(idx), 3))
    for n, i in enumerate(idx):
        _set_reference_state(data, J, ref, int(i), BALLS, HINGES, pelvis=True)
        mujoco.mj_kinematics(model, data)
        R = np.array(data.site_xmat[palm_sid]).reshape(3, 3)
        X[n] = data.site_xpos[palm_sid] + R @ local
    return ref.t[idx], X


def push_off_time(model, data, J: _Joints, ref: Reference, local: np.ndarray, palm_sid: int) -> float:
    """When the hand stops accelerating the ball: peak speed of the palm-held ball point.

    The hand can push the ball but cannot pull it back, so the ball leaves
    once the point holding it starts to slow down. Searched from 30 ms before
    to 10 ms after OBP's release event.
    """
    T, X = _ball_point_track(model, data, J, ref, local, palm_sid, ref.release - 0.030, ref.release + 0.010)
    speed = np.linalg.norm(np.gradient(X, T, axis=0), axis=1)
    return float(T[int(np.argmax(speed))])


def _held_ball(model: mujoco.MjModel, grip: Grip, ref: Reference) -> tuple[mujoco.MjData, _Joints, np.ndarray, int]:
    data = mujoco.MjData(model)
    J = _Joints(model)
    _set_reference_state(data, J, ref, 0, BALLS, HINGES, pelvis=True)
    mujoco.mj_forward(model, data)
    _fingertip_grip(model, data, grip)
    palm_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "palm")
    R = np.array(data.site_xmat[palm_sid]).reshape(3, 3)
    local = R.T @ (data.qpos[J.q["ball_free"] : J.q["ball_free"] + 3] - data.site_xpos[palm_sid])
    return data, J, local, palm_sid


def palm_alignment(motion: Motion, offsets_deg: np.ndarray, grip: Grip | str = "four-seam",
                   push_offset_deg: float | None = None) -> tuple[np.ndarray, float]:
    """Palm normal · ball direction at push-off, for each pronation offset.

    At the instant the hand stops accelerating the ball it is still pushing
    it, so the palm should be behind the ball. Returns the alignment per
    offset and the push-off time found with `push_offset_deg` (default: the
    motion's own offset, or OBP's global one).
    """
    from pitchlab.imitate import PRONATION_ZERO_OFFSET_DEG

    if isinstance(grip, str):
        grip = GRIPS[grip]
    body = BodyParams(height_m=motion.height_m, mass_kg=motion.mass_kg)
    t1 = motion.release + 0.012
    if push_offset_deg is None:
        push_offset_deg = motion.pronation_offset_deg if motion.pronation_offset_deg is not None else PRONATION_ZERO_OFFSET_DEG
    ref = _reference(motion, body, push_offset_deg, t1)
    model = build_model(grip, body, ref.geometry)
    data, J, local, palm_sid = _held_ball(model, grip, ref)
    push = push_off_time(model, data, J, ref, local, palm_sid)
    out = np.zeros(len(offsets_deg))
    for n, off in enumerate(offsets_deg):
        r = _reference(motion, body, float(off), t1)
        T, X = _ball_point_track(model, data, J, r, local, palm_sid, push - 0.002, push + 0.002, stride=1)
        k = int(np.argmin(np.abs(T - push)))
        v = np.gradient(X, T, axis=0)[k]
        _set_reference_state(data, J, r, int(np.argmin(np.abs(r.t - push))), BALLS, HINGES, pelvis=True)
        mujoco.mj_kinematics(model, data)
        normal = np.array(data.site_xmat[palm_sid]).reshape(3, 3)[:, 2]
        out[n] = float(normal @ v / np.linalg.norm(v))
    return out, push


def solve_pronation(motion: Motion, grip: Grip | str = "four-seam") -> float:
    """Pronation offset (no measurement): the palm behind the ball at push-off.

    The same criterion that calibrated OBP's global 140°, applied to a motion
    whose pronation was not measured. Coarse 15° grid, then 3° around the
    best, each with push-off recomputed at the current best.
    """
    best = 0.0
    for grid in (np.arange(-180.0, 180.0, 15.0), None):
        if grid is None:
            grid = best + np.arange(-15.0, 15.1, 3.0)
        a, _ = palm_alignment(motion, grid, grip, push_offset_deg=best)
        best = float(grid[int(np.argmax(a))])
    return best


def _as_motor(model: mujoco.MjModel, aid: int, cap: float) -> None:
    """Turn a grip servo into a pure torque source, ctrl 1 = cap (flexion)."""
    model.actuator_gainprm[aid, :3] = (cap, 0.0, 0.0)
    model.actuator_biasprm[aid, :3] = 0.0
    model.actuator_ctrlrange[aid] = (-1.0, 1.0)


def _clip_distal(name: str, tau: float) -> float:
    """Muscle ceiling. Signs are the model's positive joint directions."""
    if name == "wrist_flex":
        return float(np.clip(tau, -WRIST_EXT_LIMIT_NM, WRIST_FLEX_LIMIT_NM))
    if name == "wrist_dev":
        return float(np.clip(tau, -WRIST_RADIAL_LIMIT_NM, WRIST_ULNAR_LIMIT_NM))
    if name == "pronation":
        return float(np.clip(tau, -SUPINATION_LIMIT_NM, PRONATION_LIMIT_NM))
    if name == "elbow":
        return float(np.clip(tau, -ELBOW_EXT_LIMIT_NM, ELBOW_FLEX_LIMIT_NM))
    raise KeyError(name)


def _emg_excitation(t: float, mer_s: float) -> float:
    """0–1 command for the wrist flexors. Shape from DiGiovine 1992 phases.

    Late cocking starts at foot contact and ends at MER. Acceleration holds
    the peak until release, about 30 ms after MER in this delivery. The 40 ms
    rise before foot contact is not a published onset.
    """
    if t < -WRIST_EMG_RAMP_S:
        return 0.0
    if t < 0.0:
        return 0.25 * (t + WRIST_EMG_RAMP_S) / WRIST_EMG_RAMP_S
    if mer_s <= 1e-4 or t < mer_s:
        span = max(mer_s, 1e-3)
        return 0.25 + 0.75 * min(1.0, t / span)
    if t < mer_s + 0.040:
        return 1.0
    return max(0.0, 1.0 - (t - mer_s - 0.040) / 0.040)


def _whip_torque(name: str, q: float, qd: float, q_ref: float, t: float = 0.0, mer_s: float = 0.106) -> tuple[float, float]:
    """Passive spring, weak spring toward the published angle, and wrist EMG torque.

    Returns (unclipped sum, clipped sum). The clip is the muscle box. The
    passive term is tissue; it is small near neutral and is included in the clip.
    Wrist flexion and ulnar deviation add an open-loop torque in the muscle's
    positive direction, scaled by the EMG envelope and by WRIST_EMG_PEAK.
    """
    # Pronation's model zero is the palm-alignment offset, not anatomical
    # neutral, so the small-angle stiffness is applied about the published
    # angle. Wrist zero is the anatomical neutral the stiffness was measured
    # about. Past the AAOS end-range a ligament term rises; the muscle clip
    # still bounds the sum.
    rest = q_ref if name == "pronation" else 0.0
    passive = -PASSIVE_K_NM_PER_RAD[name] * (q - rest) - PASSIVE_B_NM_S_PER_RAD * qd
    # Ligament about the AAOS end-range. Pronation's model zero is not that
    # neutral (the grip offset sits near 138°), so it is not stopped here.
    if name != "pronation":
        lo, hi = WRIST_ROM_RAD[name]
        if q < lo:
            passive += -40.0 * (q - lo)
        elif q > hi:
            passive += -40.0 * (q - hi)
    actuator = WHIP_ACT_K_NM_PER_RAD * (q_ref - q)
    active = 0.0
    if name in WRIST_EMG_PEAK:
        # Positive model signs are flexion and ulnar deviation: FCR/FCU.
        cap = WRIST_FLEX_LIMIT_NM if name == "wrist_flex" else WRIST_ULNAR_LIMIT_NM
        active = _emg_excitation(t, mer_s) * WRIST_EMG_PEAK[name] * cap
    raw = passive + actuator + active
    return raw, _clip_distal(name, raw)


def _clip_brace_hinge(name: str, tau: float) -> float:
    """Isometric bound for a knee or an ankle. Positive knee torque is flexion."""
    if "knee" in name:
        return float(np.clip(tau, -KNEE_EXT_LIMIT_NM, KNEE_FLEX_LIMIT_NM))
    if "ankle" in name:
        return float(np.clip(tau, -ANKLE_LIMIT_NM, ANKLE_LIMIT_NM))
    raise KeyError(name)


def _geom_force(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> np.ndarray:
    """World-frame contact force on one geom, summed over its contacts."""
    total = np.zeros(3)
    for k in range(data.ncon):
        con = data.contact[k]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 != gid and g2 != gid:
            continue
        fr = np.zeros(6)
        mujoco.mj_contactForce(model, data, k, fr)
        frame = np.array(con.frame, dtype=float).reshape(3, 3)
        # mj_contactForce is the contact-frame force. With the normal pointing
        # from geom1 to geom2, frame.T @ f points geom1 into the ground during
        # compression, so the force on geom1 is the opposite of that vector.
        force = frame.T @ fr[:3]
        if g1 == gid:
            force = -force
        total += force
    return total


def _support_split(force: np.ndarray, com: np.ndarray, lead: np.ndarray, trail: np.ndarray,
                   lead_down: bool) -> tuple[np.ndarray, np.ndarray]:
    """Lead and trail share of the ground force: the lever rule on the COM's
    horizontal position between the ankles. An estimate, not a measurement."""
    if not lead_down:
        return np.zeros(3), np.array(force, dtype=float)
    axis = (lead - trail)[:2]
    span = float(axis @ axis)
    s = float(np.clip(float((com - trail)[:2] @ axis) / max(span, 1e-9), 0.0, 1.0))
    return force * s, force * (1.0 - s)


def _match_distal_qacc(model: mujoco.MjModel, data: mujoco.MjData, indexes: np.ndarray, target: np.ndarray) -> None:
    """Set `data.qacc` at `indexes` so inverse dynamics equals `target`.

    The mass matrix makes joint force linear in acceleration, so one Jacobian
    solve is exact at this pose. Accelerations outside `indexes` stay put.
    Used in the roll window: the shoulder is asked for the acceleration that
    goes with the muscle torque actually applied at the elbow and wrist, not
    for the reference acceleration those muscles cannot produce.
    """
    n = int(len(indexes))
    base_acc = data.qacc[indexes].copy()

    def forces() -> np.ndarray:
        mujoco.mj_inverse(model, data)
        return np.array([float(data.qfrc_inverse[int(j)]) for j in indexes])

    base_force = forces()
    jac = np.empty((n, n))
    for k in range(n):
        j = int(indexes[k])
        data.qacc[j] = base_acc[k] + 1.0
        jac[:, k] = forces() - base_force
        data.qacc[j] = base_acc[k]
    delta = np.linalg.solve(jac, target - base_force)
    data.qacc[indexes] = base_acc + delta


def _box_limits() -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper net torque for elbow, pronation, wrist flexion, deviation."""
    lo = np.array([-ELBOW_EXT_LIMIT_NM, -SUPINATION_LIMIT_NM, -WRIST_EXT_LIMIT_NM, -WRIST_RADIAL_LIMIT_NM])
    hi = np.array([ELBOW_FLEX_LIMIT_NM, PRONATION_LIMIT_NM, WRIST_FLEX_LIMIT_NM, WRIST_ULNAR_LIMIT_NM])
    return lo, hi


def _project_torque(jac: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Smallest acceleration change that puts `base + jac @ dx` inside the box."""
    lo, hi = _box_limits()
    if np.all(base >= lo - 1e-8) and np.all(base <= hi + 1e-8):
        return np.zeros(4)
    try:
        jinv = np.linalg.inv(jac)
    except np.linalg.LinAlgError:
        jinv = np.linalg.pinv(jac)
    metric = jinv.T @ jinv
    best: tuple[float, np.ndarray] | None = None
    for mask in np.ndindex((3, 3, 3, 3)):
        fixed = [k for k, m in enumerate(mask) if m]
        free = [k for k, m in enumerate(mask) if not m]
        dtau = np.zeros(4)
        for k in fixed:
            dtau[k] = (lo[k] if mask[k] == 1 else hi[k]) - base[k]
        if free:
            block = metric[np.ix_(free, free)]
            rhs = -metric[np.ix_(free, fixed)] @ dtau[fixed] if fixed else np.zeros(len(free))
            try:
                dtau[free] = np.linalg.solve(block, rhs)
            except np.linalg.LinAlgError:
                continue
            if np.any(dtau[free] < lo[free] - base[free] - 1e-5):
                continue
            if np.any(dtau[free] > hi[free] - base[free] + 1e-5):
                continue
        dx = jinv @ dtau
        grad = metric @ dtau
        # grad = d(cost)/d(dtau). A lower bound is optimal when the gradient
        # points upward (λ ≥ 0); an upper bound when it points downward.
        if any((mask[k] == 1 and grad[k] < -1e-5) or (mask[k] == 2 and grad[k] > 1e-5) for k in fixed):
            continue
        cost = float(dx @ dx)
        if best is None or cost < best[0]:
            best = (cost, dx)
    if best is None:
        return jinv @ (np.clip(base, lo, hi) - base)
    return best[1]


def _quintic_error(p0: np.ndarray, v0: np.ndarray, t: float, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position, velocity and acceleration of an error that is gone after `duration`.

    Starts at `p0`, `v0` with zero acceleration and ends at rest. The zero
    end-acceleration keeps the hand from a one-sample torque spike where it
    rejoins the published curve.
    """
    if duration <= 1e-6:
        return p0, v0, np.zeros_like(p0)
    u = float(np.clip(t / duration, 0.0, 1.0))
    u2, u3, u4, u5 = u * u, u ** 3, u ** 4, u ** 5
    p = p0 * (1.0 - 10.0 * u3 + 15.0 * u4 - 6.0 * u5) + (v0 * duration) * (u - 6.0 * u3 + 8.0 * u4 - 3.0 * u5)
    dp = p0 * (-30.0 * u2 + 60.0 * u3 - 30.0 * u4) + (v0 * duration) * (1.0 - 18.0 * u2 + 32.0 * u3 - 15.0 * u4)
    ddp = p0 * (-60.0 * u + 180.0 * u2 - 120.0 * u3) + (v0 * duration) * (-36.0 * u + 96.0 * u2 - 60.0 * u3)
    return p, dp / duration, ddp / (duration * duration)


def _quat_exp(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    half = 0.5 * angle
    axis = rotvec / angle
    out = np.empty(4)
    out[0] = np.cos(half)
    out[1:] = axis * np.sin(half)
    return out


def _feasible_release(model: mujoco.MjModel, data: mujoco.MjData, J: _Joints, ref: Reference, finger_hold: list[tuple[int, int, float]]) -> None:
    """Rebuild release acceleration so the unloaded torque fits the muscle box.

    The wrist's own angular acceleration contributes under 1 N·m. The 60 N·m
    spike is the shoulder's spin climbing after kinematic push-off, while the
    ball is still on the fingers. From 9 ms before that push until 10 ms
    after, elbow acceleration and shoulder angular acceleration are replaced
    by the smallest change whose inverse dynamics lies inside the elbow,
    pronation and wrist limits. Push-off moves when the acceleration changes,
    so the window is placed again. The arm then returns to the published
    curve over 20 ms; leaving it on the exit velocity pointed the hand away
    from the plate.
    """
    dt = float(ref.t[1] - ref.t[0])
    inv = copy.deepcopy(model)
    inv.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    inv.actuator_gainprm[:] = 0.0
    inv.actuator_biasprm[:] = 0.0
    feed = mujoco.MjData(inv)
    names = ("elbow", "pronation", "wrist_flex", "wrist_dev")
    pv = J.v["pelvis_free"]

    def torques(i: int, elbow_qdd: float, gh_a: np.ndarray) -> np.ndarray:
        _set_reference_state(feed, J, ref, i, BALLS, HINGES, pelvis=True)
        for q_adr, v_adr, init in finger_hold:
            feed.qpos[q_adr] = init
            feed.qvel[v_adr] = 0.0
        feed.qacc[:] = 0.0
        feed.qacc[pv : pv + 3] = ref.pelvis["acc"][i]
        feed.qacc[pv + 3 : pv + 6] = ref.pelvis["a"][i]
        for name in BALLS:
            feed.qacc[J.v[name] : J.v[name] + 3] = ref.ball[name][2][i]
        for name in HINGES:
            feed.qacc[J.v[name]] = float(ref.hinge[name][2][i])
        feed.qacc[J.v["elbow"]] = elbow_qdd
        feed.qacc[J.v["gh"] : J.v["gh"] + 3] = gh_a
        mujoco.mj_inverse(inv, feed)
        return np.array([float(feed.qfrc_inverse[J.v[n]]) for n in names])

    q, qd, qdd = ref.hinge["elbow"]
    qq, w, a = ref.ball["gh"]
    saved = (q.copy(), qd.copy(), qdd.copy(), qq.copy(), w.copy(), a.copy())
    palm_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "palm")
    # The ball is already seated in the grip. Measuring the held point from
    # the reference pose (the ball back on the wrist path) moves push-off
    # and the torque window misses the fingers.
    pose = (data.qpos.copy(), data.qvel.copy())
    mujoco.mj_kinematics(model, data)
    palm_R = np.array(data.site_xmat[palm_sid]).reshape(3, 3)
    bq = J.q["ball_free"]
    local = palm_R.T @ (data.qpos[bq : bq + 3] - data.site_xpos[palm_sid])

    def _push() -> float:
        data.qpos[:], data.qvel[:] = pose
        return push_off_time(model, data, J, ref, local, palm_sid)

    def _restore() -> None:
        q[:], qd[:], qdd[:], qq[:], w[:], a[:] = (x.copy() for x in saved)

    def _solve(t_lo: float, t_hi: float) -> None:
        i0 = max(int(np.searchsorted(ref.t, t_lo)), 1)
        i1 = min(int(np.searchsorted(ref.t, t_hi)), len(ref.t) - 2)
        if i1 <= i0 + 5:
            return
        jac = None
        for i in range(i0, i1):
            e0 = float(qdd[i])
            g0 = a[i].copy()
            base = torques(i, e0, g0)
            if jac is None or (i - i0) % 20 == 0:
                jac = np.zeros((4, 4))
                jac[:, 0] = torques(i, e0 + 1.0, g0) - base
                for k in range(3):
                    g = g0.copy()
                    g[k] += 1.0
                    jac[:, k + 1] = torques(i, e0, g) - base
                base = torques(i, e0, g0)
            delta = _project_torque(jac, base)
            e_new = e0 + float(delta[0])
            g_new = g0 + delta[1:]
            qdd[i] = e_new
            a[i] = g_new
            qd[i + 1] = qd[i] + dt * e_new
            q[i + 1] = q[i] + dt * qd[i]
            w[i + 1] = w[i] + dt * g_new
            ww = np.array([0.0, w[i, 0], w[i, 1], w[i, 2]])
            qq[i + 1] = qq[i] + dt * 0.5 * quat_mul(qq[i], ww)
            nrm = float(np.linalg.norm(qq[i + 1]))
            qq[i + 1] /= nrm if nrm > 1e-12 else 1.0
        # Return to the published curve. The error is a few degrees and a few
        # tens of rad/s; a 20 ms quintic brings it back without a kink.
        blend = 0.020
        i2 = min(i1 + int(round(blend / dt)), len(ref.t) - 1)
        duration = (i2 - i1) * dt
        q_err0 = float(q[i1] - saved[0][i1])
        qd_err0 = float(qd[i1] - saved[1][i1])
        rot0 = quat_log(quat_mul(quat_conj(saved[3][i1]), qq[i1]))
        w_err0 = w[i1] - saved[4][i1]
        for i in range(i1, i2 + 1):
            t_rel = (i - i1) * dt
            eq, ev, ea = _quintic_error(np.array([q_err0]), np.array([qd_err0]), t_rel, duration)
            q[i] = saved[0][i] + float(eq[0])
            qd[i] = saved[1][i] + float(ev[0])
            qdd[i] = saved[2][i] + float(ea[0])
            rot, wv, wa = _quintic_error(rot0, w_err0, t_rel, duration)
            qq[i] = quat_mul(saved[3][i], _quat_exp(rot))
            w[i] = saved[4][i] + wv
            a[i] = saved[5][i] + wa

    push = _push()
    for _ in range(3):
        _restore()
        # 9 ms before push is the roll. 10 ms after covers the ball still on
        # the fingers (about 7 ms) and the acceleration the spline adds there.
        _solve(push - 0.009, push + 0.010)
        nxt = _push()
        if abs(nxt - push) < 0.002:
            push = nxt
            break
        push = nxt
    data.qpos[:], data.qvel[:] = pose


def _contact_probe(model, data, t, bq, bv, ball_bid) -> dict:
    leather = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "leather")
    center = data.qpos[bq : bq + 3].copy()
    forces: dict[str, np.ndarray] = {}
    detail: list[tuple[str, np.ndarray, np.ndarray]] = []
    f6 = np.zeros(6)
    for k in range(data.ncon):
        con = data.contact[k]
        g1, g2 = int(con.geom1), int(con.geom2)
        n1 = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or "")
        n2 = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or "")
        if leather >= 0 and int(model.geom_bodyid[g1]) != int(model.geom_bodyid[g2]):
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            ball_body = int(model.geom_bodyid[leather])
            if ball_body in (b1, b2):
                mujoco.mj_contactForce(model, data, k, f6)
                frame = np.array(con.frame).reshape(3, 3)
                on_other = frame.T @ f6[:3]
                other = n2 if b1 == ball_body else n1
                if b2 == ball_body:
                    on_other = -on_other
                detail.append((other or f"geom{g2 if b1 == ball_body else g1}", on_other, np.array(con.pos) - center))
        if _finger_of_pad(n1) and not _finger_of_pad(n2):
            pad_geom, pad_name = g1, n1
        elif _finger_of_pad(n2) and not _finger_of_pad(n1):
            pad_geom, pad_name = g2, n2
        else:
            continue
        finger = _finger_of_pad(pad_name)
        if finger is None or leather < 0:
            continue
        mujoco.mj_contactForce(model, data, k, f6)
        frame = np.array(con.frame).reshape(3, 3)
        # Force on the pad. mj_contactForce acts on geom1.
        world = frame.T @ f6[:3]
        if pad_geom == g2:
            world = -world
        forces[finger] = forces.get(finger, np.zeros(3)) + world
    pads = {}
    for f in (*FINGERS, "palm"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{f}_pad" if f != "palm" else "palm")
        pads[f] = data.site_xpos[sid] - center
    return {
        "t": t,
        "v": data.qvel[bv : bv + 3].copy(),
        "w": _rot(data, ball_bid) @ data.qvel[bv + 3 : bv + 6],
        "forces": forces,
        "pads": pads,
        "detail": detail,
    }


def _finger_force_parts(trace: list[tuple[float, str, np.ndarray, np.ndarray]], release_t: float) -> dict[str, dict[str, float]]:
    """Split each finger's contact force into the pieces Kinoshita publishes.

    Kinoshita, Obata, Nasu, Kadota, Matsuo, Fleisig, Hum Mov Sci (2017): the
    quoted number is the second peak of the resultant, about 8 ms before
    release, and shear is about half of that. The sampling filter is not in
    the abstract we use, so two numbers are both reported. `raw_max_n` is the
    unfiltered maximum from the start of the roll until release. `window_max_n`
    is the maximum in the 15-to-4 ms window before release (the second-peak
    window; 4 ms is the edge of that window and is assumed). Normal is the
    radial part, positive when the ball pushes the pad off its surface.
    """
    by: dict[str, list[tuple[float, float, float, float]]] = {}
    for t, name, force, pad in trace:
        rel_ms = (t - release_t) * 1000.0
        resultant = float(np.linalg.norm(force))
        radial = pad / max(float(np.linalg.norm(pad)), 1e-9)
        normal = float(force @ radial)
        shear = math.sqrt(max(0.0, resultant * resultant - normal * normal))
        by.setdefault(name, []).append((rel_ms, resultant, normal, shear))
    out = {}
    for name, rows in by.items():
        raw = max(rows, key=lambda row: row[1])
        window = [row for row in rows if -15.0 <= row[0] <= -4.0]
        picked = max(window, key=lambda row: row[1]) if window else raw
        near = min(rows, key=lambda row: abs(row[0] + 8.0))
        # Kinoshita's ISBS 2014 precursor (same USL06-H5-500N-C transducer in
        # the ball) sampled each channel at 1 kHz and did not state a low-pass.
        # One sample per millisecond is that measurement. A cutoff is not invented.
        on_ms: dict[int, tuple[float, float, float, float]] = {}
        for row in rows:
            key = int(round(row[0]))
            prev = on_ms.get(key)
            if prev is None or abs(row[0] - key) < abs(prev[0] - key):
                on_ms[key] = row
        khz = [on_ms[k] for k in sorted(on_ms) if -15 <= k <= -4]
        khz_peak = max(khz, key=lambda row: row[1]) if khz else raw
        out[name] = {
            "series_ms": [round(row[0], 2) for row in rows],
            "series_n": [round(row[1], 1) for row in rows],
            "series_normal": [round(row[2], 1) for row in rows],
            "series_shear": [round(row[3], 1) for row in rows],
            "raw_max_n": raw[1],
            "raw_max_ms": raw[0],
            "window_max_n": picked[1],
            "window_max_ms": picked[0],
            "window_normal_n": picked[2],
            "window_shear_n": picked[3],
            "at_m8_n": near[1],
            "at_m8_ms": near[0],
            "at_m8_normal_n": near[2],
            "at_m8_shear_n": near[3],
            "khz_max_n": khz_peak[1],
            "khz_max_ms": khz_peak[0],
            "khz_normal_n": khz_peak[2],
            "khz_shear_n": khz_peak[3],
            "khz_series_n": [round(on_ms[k][1], 1) for k in sorted(on_ms) if -16 <= k <= 1],
            "khz_series_ms": [k for k in sorted(on_ms) if -16 <= k <= 1],
        }
    return out


def simulate_delivery(
    grip: Grip | str = "four-seam",
    motion: Motion | None = None,
    body: BodyParams | None = None,
    store_frames: bool = False,
    frame_dt: float = 0.002,
    friction: float | None = None,
    probe: list | None = None,
    roll_start_s: float | None = None,
    roll_delay_s: float = 0.0,
    carry_spin: str = "palm",
    torque_scale: float = 1.0,
    roll_mode: str = "flex",
    model_hook=None,
    seam_yaw_deg: float = 0.0,
    seam_tilt_deg: float = 0.0,
    pulp: bool = True,
    ball=None,
    carry_allow_m: float | None = None,
    distal_forward: bool = True,
    body_mode: str = "braced",
):
    """Integrate one delivery. Default motion: the calibrated single pitcher (no dataset).

    `probe`, if given, collects contact forces, pad positions, and finger
    angles every 1 ms. `roll_start_s` overrides when the grip opens (default:
    CONTACT_MS before push-off); the outcome is sensitive to it at the 1 ms
    level, as Hore et al. (1995) found for skilled throwers.

    `carry_spin`:
      "palm" — the ball's angular velocity is the palm's (rigid weld).
      "zero" — translation follows the palm, but angular velocity is held at
        0 until the roll. Yamaguchi, Suzuki, Nishi, Fukuda, Nasu et al.,
        Sci Rep (2025): group-mean ball ω is ~0 rad/s at thumb-off (t = −8 ms),
        then rises through the fingertip roll. Copying palm ω plants topspin
        that those pitchers do not show.
    `torque_scale` multiplies the rolling fingers' torque cap.
    `roll_mode`: "flex" curls the rolling fingers at the cap (default);
    "hold" leaves them on the grip servo; "extend" extends PIP and DIP
    (Matsuo 2018) while the MCP still flexes.
    `model_hook(model)` runs once after the model is built (contact experiments).
    `pulp` adds the compliant nodes on the index and middle. It is on: the
    four-seam grip puts a seam under those pads, and the nodes are what can
    hook the ridge (NPB standard 0.90 mm; the MLB preset is 0.80 mm).
    `ball` is "npb" (default) or "mlb". `friction` overrides the leather
    coefficient of that ball and scales the seam coefficient with it. The
    default None uses the ball's own measured μ.
    `seam_yaw_deg` / `seam_tilt_deg` are extra turns after the grip is closed,
    about the palm normal and the index–middle line. The default of 0 leaves
    the ball where `fourseam.four_seam_rotation` placed it. They are not a
    measured pose.
    `distal_forward`: from foot contact, pronation and both wrist hinges are a
    passive spring plus a weak spring toward the published angle, clipped to
    isometric strength, so the hand lags the reference. In the roll window the
    elbow is a capped servo toward that angle (no reference-acceleration
    feedforward) and the fingers add a voluntary flexion at their ceiling.
    Shoulder and trunk stay on the tracking servo, at the acceleration those
    torques produce. After foot contact the pelvis translation is not forced.
    `body_mode`: see BODY_MODES. The wrist, pronation, elbow roll window and
    fingers are the same in both modes.
    """
    if body_mode not in BODY_MODES:
        raise ValueError(f"body_mode must be one of {BODY_MODES}, got {body_mode!r}")
    guided = body_mode == "guided"
    if isinstance(grip, str):
        grip = GRIPS[grip]
    if motion is None:
        motion = default_motion()
    body, ref = prepare(motion, body, grip)
    model = build_model(grip, body, ref.geometry, friction, pulp=pulp, ball=ball)
    if model_hook is not None:
        model_hook(model)
    data = mujoco.MjData(model)
    J = _Joints(model)
    inverse_model = copy.deepcopy(model)
    inverse_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    # Finger servos must not leak into qfrc_inverse. The torque we keep is the
    # joint force that produces the commanded acceleration with no actuators.
    inverse_model.actuator_gainprm[:] = 0.0
    inverse_model.actuator_biasprm[:] = 0.0
    scratch = mujoco.MjData(inverse_model)
    # Unloaded reference: inverse dynamics of the reference pose, with no PD
    # and no ball. Its joint torque is what gets applied in the roll window.
    feed = mujoco.MjData(inverse_model)
    steps = len(ref.t) - 1
    wn = 2.0 * math.pi * SERVO_HZ
    kp, kd = wn * wn, 2.0 * wn
    wn_leg = 2.0 * math.pi * LEG_SERVO_HZ
    kp_leg, kd_leg = wn_leg * wn_leg, 2.0 * wn_leg
    leather_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "leather")
    leather_con = (int(model.geom_contype[leather_gid]), int(model.geom_conaffinity[leather_gid]))
    foot_gids = {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot") for side in ("lead", "trail")
    }
    if guided:
        for gid in foot_gids.values():
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0
    ankle_sids = {side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ankle") for side in ("lead", "trail")}
    pelvis_bid = _body(model, "pelvis")
    leg_dofs = np.array(
        [J.v[n] + k for n in KINEMATIC_BALLS for k in range(3)] + [J.v[n] for n in KINEMATIC_HINGES], dtype=int
    )
    def _set_leather(on: bool) -> None:
        model.geom_contype[leather_gid] = leather_con[0] if on else 0
        model.geom_conaffinity[leather_gid] = leather_con[1] if on else 0

    _set_reference_state(data, J, ref, 0, BALLS, HINGES, pelvis=True)
    mujoco.mj_forward(model, data)
    _fingertip_grip(model, data, grip)
    # Retargeting shoulder acceleration so the wrist torque is zero also turns
    # the arm off the plate (launch went to +14° and +167°). The reference
    # keeps the published curve. The ceiling below is what the muscles apply.
    if distal_forward and FEASIBLE_RELEASE:
        held = []
        for finger in FINGERS:
            for joint in ("mcp", "pip", "dip"):
                jid = _jid(model, f"{finger}_{joint}")
                q_adr, v_adr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
                held.append((q_adr, v_adr, float(data.qpos[q_adr])))
        _feasible_release(model, data, J, ref, held)
    if seam_yaw_deg or seam_tilt_deg:
        _turn_ball(model, data, seam_yaw_deg, seam_tilt_deg)
        mujoco.mj_forward(model, data)
        _seat_pulp(model, data)
    fingers = []
    for finger in FINGERS:
        for joint in ("mcp", "pip", "dip"):
            jid = _jid(model, f"{finger}_{joint}")
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{finger}_{joint}_act")
            q_adr, v_adr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
            fingers.append((finger, joint, aid, q_adr, v_adr, float(data.qpos[q_adr])))
    rolling = {aid: False for _, _, aid, _, _, _ in fingers}

    palm_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "palm")
    pad_ids = [
        (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{finger}_pad"), BALL_RADIUS_M + PAD_RADIUS_M)
        for finger in FINGERS
    ]
    belly_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{finger}_pad") for finger in ("index", "middle")
    ]
    ball_bid = _body(model, "ball")
    R_palm = np.array(data.site_xmat[palm_sid]).reshape(3, 3)
    bq = J.q["ball_free"]
    bv = J.v["ball_free"]
    carry_local = R_palm.T @ (data.qpos[bq : bq + 3] - data.site_xpos[palm_sid])
    carry_rot = R_palm.T @ _rot(data, ball_bid)
    state = (data.qpos.copy(), data.qvel.copy())
    release = push_off_time(model, data, J, ref, carry_local, palm_sid)
    data.qpos[:], data.qvel[:] = state
    mujoco.mj_forward(model, data)
    carry_until = release - ROLL_BEFORE_PUSH_S if roll_start_s is None else roll_start_s
    carry_until += roll_delay_s
    # Stevenson (1985): the thumb leaves ~6 ms before the ball, which leaves
    # the fingers ~CONTACT_MS after the roll starts (Matsuo et al. 2018).
    open_at = carry_until + (CONTACT_MS - THUMB_OFF_MS) / 1000.0

    bids = {n: _body(model, n) for n in ("pelvis", "upper_trunk", "humerus", "forearm", "hand")}
    sid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("shoulder", "elbow", "wrist", "head", "thorax_prox")}

    frames: list[np.ndarray] = []
    next_frame = ref.t[0]
    seen = False
    quiet = 0
    quiet_need = int(round(0.002 / TIMESTEP_S))
    release_step = None
    release_velocity = release_omega = release_pos = None
    min_finger = 1e9
    sq_err = 0.0
    n_err = 0
    track_sq: dict[str, float] = {}
    track_n: dict[str, int] = {}
    track_max: dict[str, tuple[float, float]] = {}
    hand_err = []
    keys = ("t", "pelvis_w", "trunk_w", "elbow_w", "ir_w", "wrist_w", "sep", "fer", "fwd", "lat", "rot",
            "ir_moment", "varus", "sh_comp", "el_comp", "elbow_deg", "ball_wx", "ball_wy", "ball_wz", "contacts",
            "pelvis_vy", "lead_knee_deg", "grf_lead_y", "grf_lead_z", "grf_trail_y", "grf_trail_z")
    log: dict[str, list[float]] = {k: [] for k in keys}
    contributions: dict[str, np.ndarray] = {}
    sampled = False
    hand_speed = 0.0
    finger_force: dict[str, float] = {}
    finger_trace: list[tuple[float, str, np.ndarray, np.ndarray]] = []
    distal_raw: dict[str, list[float]] = {n: [] for n in YIELD_HINGES}
    distal_applied: dict[str, list[float]] = {n: [] for n in YIELD_HINGES}
    distal_ceiling: dict[str, list[float]] = {n: [] for n in YIELD_HINGES}
    wrist_vs_ref: list[tuple[float, float]] = []
    whip_err: dict[str, list[tuple[float, float, float]]] = {n: [] for n in WHIP_HINGES}
    finger_limit = {"mcp": MCP_TORQUE_NM, "pip": PIP_TORQUE_NM, "dip": DIP_TORQUE_NM}
    contact_off = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    block_latched = False
    late_snapped = False
    Lgeom = ref.geometry.lengths
    plant_prev: dict[str, float] = {}
    plant_quat_prev: dict[str, np.ndarray] = {}

    t_start = float(os.environ.get("PITCHLAB_T_START", "-1e9"))
    for i in range(steps):
        t = float(ref.t[i])
        if t < t_start:
            continue
        # A shortened run still has to start on the published pose. Otherwise
        # the arm is the compiled zero and the landing torque is fiction.
        if not late_snapped and t_start > float(ref.t[0]) + 1e-6:
            _set_reference_state(data, J, ref, i, BALLS, HINGES, pelvis=True)
            late_snapped = True
        # Until foot contact the pelvis and the legs are the published pose.
        # After that, translation is left to contact and the legs are a servo.
        # The first dynamic step starts from that pose, not from one integrated
        # step taken with the ground turned off.
        block = t >= -1e-12
        braced_now = block and not guided
        if guided:
            _set_reference_state(data, J, ref, i, KINEMATIC_BALLS, KINEMATIC_HINGES, pelvis=True)
        elif not block or not block_latched:
            _set_reference_state(data, J, ref, i, KINEMATIC_BALLS, KINEMATIC_HINGES, pelvis=True)
            block_latched = block
        plant_q: dict[str, np.ndarray] = {}
        plant_w: dict[str, np.ndarray] = {}
        plant_h: dict[str, float] = {}
        plant_hd: dict[str, float] = {}
        plant_watch: dict[str, tuple[float, float, float]] = {}
        if braced_now:
            R_pel = _rot(data, bids["pelvis"])
            p_pos = np.array(data.qpos[J.q["pelvis_free"] : J.q["pelvis_free"] + 3], dtype=float)
            for side, sign, ankle_key, knee_key in (
                ("lead", -1.0, "lead_ankle_jc", "lead_knee_jc"),
                ("trail", 1.0, "rear_ankle_jc", "rear_knee_jc"),
            ):
                ankle_w = np.array(ref.points[ankle_key][i], dtype=float)
                ankle_w[2] -= PLANT_PRESS_M
                hip_w = p_pos + R_pel @ np.array([sign * float(Lgeom["hip_half"]), 0.0, 0.0])
                plant_watch[side] = (float(hip_w[2]), float(ankle_w[2]), float(np.linalg.norm(hip_w - ankle_w)))
                q_hip, knee, ankle = _reach_plant(
                    side,
                    sign,
                    float(Lgeom["hip_half"]),
                    float(Lgeom["thigh"]),
                    float(Lgeom["shank"]),
                    p_pos,
                    R_pel,
                    ankle_w,
                    np.array(ref.points[knee_key][i], dtype=float),
                )
                prev_q = plant_quat_prev.get(f"{side}_hip")
                if prev_q is not None and float(np.dot(q_hip, prev_q)) < 0.0:
                    q_hip = -q_hip
                plant_q[f"{side}_hip"] = q_hip
                if prev_q is None:
                    plant_w[f"{side}_hip"] = np.zeros(3)
                else:
                    plant_w[f"{side}_hip"] = quat_log(quat_mul(quat_conj(prev_q), q_hip)) / TIMESTEP_S
                plant_quat_prev[f"{side}_hip"] = np.array(q_hip, dtype=float)
                for hinge, angle in ((f"{side}_knee", knee), (f"{side}_ankle", ankle)):
                    plant_h[hinge] = angle
                    prev = plant_prev.get(hinge)
                    plant_hd[hinge] = 0.0 if prev is None else (angle - prev) / TIMESTEP_S
                    plant_prev[hinge] = angle
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = data.qvel
        scratch.qacc[:] = 0.0
        pv = J.v["pelvis_free"]
        pq = J.q["pelvis_free"]
        if braced_now:
            # Leave the pelvis acceleration at 0 in the inverse. Copying the
            # contact spike back in makes the trunk torque cancel the ground,
            # and the next step blows up. The applied pelvis force stays 0
            # for translation; rotation is a separate clipped PD couple.
            q_pel = data.qpos[pq + 3 : pq + 7]
            err = quat_log(quat_mul(quat_conj(q_pel), ref.pelvis["quat"][i]))
            sq_err += float(err @ err)
            n_err += 1
        else:
            scratch.qacc[pv : pv + 3] = ref.pelvis["acc"][i]
            scratch.qacc[pv + 3 : pv + 6] = ref.pelvis["a"][i]
        for name in BALLS:
            q_ref, w_ref, a_ref = ref.ball[name]
            if braced_now and name in KINEMATIC_BALLS:
                q = data.qpos[J.q[name] : J.q[name] + 4]
                q_tgt = plant_q.get(name, q_ref[i])
                w_tgt = plant_w.get(name, np.zeros(3))
                err = quat_log(quat_mul(quat_conj(q), q_tgt))
                w = data.qvel[J.v[name] : J.v[name] + 3]
                # Position hold only. Feedforward acceleration drives the foot
                # through the mound and the contact throws the body up.
                # The plant quaternion moves as the pelvis moves. Damping
                # toward zero fights that, so the rate reference is the
                # target's own rate.
                acc = kp_leg * err + kd_leg * (w_tgt - w)
                nrm = float(np.linalg.norm(acc))
                if nrm > 40.0:
                    acc *= 40.0 / nrm
                scratch.qacc[J.v[name] : J.v[name] + 3] = acc
                continue
            acc = a_ref[i].copy()
            if name in DRIVEN_BALLS:
                q = data.qpos[J.q[name] : J.q[name] + 4]
                err = quat_log(quat_mul(quat_conj(q), q_ref[i]))
                w = data.qvel[J.v[name] : J.v[name] + 3]
                kpp, kdd = (kp_leg, kd_leg) if braced_now else (kp, kd)
                acc += kpp * err + kdd * (w_ref[i] - w)
                sq_err += float(err @ err)
                n_err += 1
                track_sq[name] = track_sq.get(name, 0.0) + float(err @ err)
                track_n[name] = track_n.get(name, 0) + 1
                track_max[name] = max(track_max.get(name, (0.0, 0.0)), (float(np.linalg.norm(err)), t))
            scratch.qacc[J.v[name] : J.v[name] + 3] = acc
        # From foot contact the wrist and pronation are not asked to track.
        # The roll window also drops the elbow's reference acceleration.
        yield_now = bool(distal_forward and release_step is None and t + 1e-12 >= carry_until)
        whip_now = bool(distal_forward and block)
        for name in HINGES:
            q_ref, qd_ref, qdd_ref = ref.hinge[name]
            if braced_now and name in KINEMATIC_HINGES:
                q_tgt_h = plant_h.get(name, float(q_ref[i]))
                qd_tgt_h = plant_hd.get(name, float(qd_ref[i]))
                e = float(q_tgt_h - data.qpos[J.q[name]])
                acc = kp_leg * e + kd_leg * (qd_tgt_h - data.qvel[J.v[name]])
                scratch.qacc[J.v[name]] = float(np.clip(acc, -40.0, 40.0))
            elif whip_now and name in WHIP_HINGES:
                # Zero, not the reference acceleration. The inverse then does
                # not ask the shoulder to drag the hand through that acceleration.
                scratch.qacc[J.v[name]] = 0.0
            elif yield_now and name == "elbow":
                e = float(q_ref[i] - data.qpos[J.q[name]])
                scratch.qacc[J.v[name]] = kp_leg * e + kd_leg * (float(qd_ref[i]) - data.qvel[J.v[name]])
            else:
                acc = float(qdd_ref[i])
                if name in DRIVEN_HINGES and not (yield_now and name in YIELD_HINGES):
                    e = float(q_ref[i] - data.qpos[J.q[name]])
                    kpp, kdd = (kp_leg, kd_leg) if braced_now else (kp, kd)
                    acc += kpp * e + kdd * (float(qd_ref[i]) - data.qvel[J.v[name]])
                    sq_err += e * e
                    n_err += 1
                    track_sq[name] = track_sq.get(name, 0.0) + e * e
                    track_n[name] = track_n.get(name, 0) + 1
                    track_max[name] = max(track_max.get(name, (0.0, 0.0)), (abs(e), t))
                scratch.qacc[J.v[name]] = acc
        # Whip torque is a spring, not the inverse dynamics of the reference
        # acceleration. Matching the shoulder acceleration to that small torque
        # keeps the humerus from being asked to drag a wrist it cannot move.
        inv_fingers = scratch
        applied_distal: dict[str, float] = {}
        if whip_now:
            for name in WHIP_HINGES:
                q = float(data.qpos[J.q[name]])
                qd = float(data.qvel[J.v[name]])
                q_ref_n = float(ref.hinge[name][0][i])
                raw, applied = _whip_torque(name, q, qd, q_ref_n, t, float(ref.events["MER_time"]))
                applied_distal[name] = applied
                if release_step is None:
                    distal_raw[name].append(raw)
                    distal_applied[name].append(applied)
                    distal_ceiling[name].append(applied)
                    whip_err[name].append((t, q, q_ref_n))
        if yield_now:
            q_w = float(data.qpos[J.q["wrist_flex"]])
            wrist_vs_ref.append((q_w, float(ref.hinge["wrist_flex"][0][i])))
            # Finger torque is the unloaded grip, not the contact-perturbed angle.
            feed.qpos[:] = data.qpos
            feed.qvel[:] = data.qvel
            _set_reference_state(feed, J, ref, i, BALLS, HINGES, pelvis=True)
            for _, _, _, q_adr, v_adr, init in fingers:
                feed.qpos[q_adr] = init
                feed.qvel[v_adr] = 0.0
            feed.qacc[:] = 0.0
            feed.qacc[pv : pv + 3] = ref.pelvis["acc"][i]
            feed.qacc[pv + 3 : pv + 6] = ref.pelvis["a"][i]
            for name in BALLS:
                feed.qacc[J.v[name] : J.v[name] + 3] = ref.ball[name][2][i]
            for name in HINGES:
                feed.qacc[J.v[name]] = float(ref.hinge[name][2][i])
            mujoco.mj_inverse(inverse_model, feed)
            inv_fingers = feed
        if guided:
            # The arm's published accelerations reach 13 000 rad/s² at the
            # shoulder and 6 000 at the elbow; clipping them slows the hand.
            # The legs are placed, not driven, and the trail leg has a
            # one-sample length-clamp glitch at foot contact (2.9e5 rad/s²)
            # that would only show up as a spike in the root force.
            scratch.qacc[leg_dofs] = np.clip(scratch.qacc[leg_dofs], -1500.0, 1500.0)
        else:
            np.clip(scratch.qacc, -1500.0, 1500.0, out=scratch.qacc)
        mujoco.mj_inverse(inverse_model, scratch)
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[J.skeleton] = scratch.qfrc_inverse[J.skeleton]
        root_force = scratch.qfrc_inverse[pv : pv + 3].copy()
        if braced_now:
            for name in KINEMATIC_HINGES:
                q_ref, qd_ref, _ = ref.hinge[name]
                q_tgt_h = plant_h.get(name, float(q_ref[i]))
                qd_tgt_h = plant_hd.get(name, float(qd_ref[i]))
                err_h = float(q_tgt_h - data.qpos[J.q[name]])
                tau_h = kp_leg * err_h + kd_leg * (qd_tgt_h - float(data.qvel[J.v[name]]))
                data.qfrc_applied[int(J.v[name])] = _clip_brace_hinge(name, tau_h)
            # Both hips, every step. The contact-free inverse torque is what
            # the leg would need with the ground turned off; leaving it on the
            # lead hip makes that couple travel up into the shoulder.
            brace = 1.0
            hip_cap = HIP_LIMIT_NM * brace
            for name in KINEMATIC_BALLS:
                q_ref, w_ref, _ = ref.ball[name]
                q = data.qpos[J.q[name] : J.q[name] + 4]
                q_tgt = plant_q.get(name, q_ref[i])
                w_tgt = plant_w.get(name, np.zeros(3))
                err_b = quat_log(quat_mul(quat_conj(q), q_tgt))
                w = data.qvel[J.v[name] : J.v[name] + 3]
                tau_b = kp_leg * err_b + kd_leg * (w_tgt - w)
                hip_n = float(np.linalg.norm(tau_b))
                if hip_n > hip_cap:
                    tau_b = tau_b * (hip_cap / hip_n)
                v0 = int(J.v[name])
                data.qfrc_applied[v0 : v0 + 3] = tau_b
            data.qfrc_applied[pv : pv + 3] = 0.0
            q_pel = data.qpos[pq + 3 : pq + 7]
            err_pel = quat_log(quat_mul(quat_conj(q_pel), ref.pelvis["quat"][i]))
            w_pel = data.qvel[pv + 3 : pv + 6]
            tau_pel = kp_leg * err_pel + kd_leg * (ref.pelvis["w"][i] - w_pel)
            n_pel = float(np.linalg.norm(tau_pel))
            pel_cap = PELVIS_ANG_LIMIT_NM * brace
            if n_pel > pel_cap:
                tau_pel *= pel_cap / n_pel
            data.qfrc_applied[pv + 3 : pv + 6] = tau_pel
        for name, tau in applied_distal.items():
            data.qfrc_applied[int(J.v[name])] = tau
        if yield_now:
            elbow_i = int(J.v["elbow"])
            elbow_raw = float(scratch.qfrc_inverse[elbow_i])
            elbow_applied = _clip_distal("elbow", elbow_raw)
            data.qfrc_applied[elbow_i] = elbow_applied
            applied_distal["elbow"] = elbow_applied
            distal_raw["elbow"].append(elbow_raw)
            distal_applied["elbow"].append(elbow_applied)
            distal_ceiling["elbow"].append(elbow_applied)
        # Fingers are torque-driven. Until the roll they hold the grip. In the
        # roll window the torque is the contact-free inverse dynamics of a
        # held joint (qacc = 0) plus a voluntary flexion or extension at the
        # literature ceiling. The sum cannot exceed that ceiling, so the ball
        # can push a finger open. Outside that window the old path remains:
        # rolling fingers flex at the scaled cap, the others let go.
        for f, j, aid, q_adr, v_adr, init in fingers:
            if yield_now:
                limit = finger_limit[j] * torque_scale
                if limit <= 0.0:
                    data.ctrl[aid] = 0.0
                    continue
                if not rolling[aid]:
                    _as_motor(model, aid, limit)
                    rolling[aid] = True
                if release_step is not None:
                    data.ctrl[aid] = 0.0
                    continue
                tau_id = float(inv_fingers.qfrc_inverse[v_adr])
                rolling_this = f in grip.roll_fingers and roll_mode != "hold"
                if roll_mode in ("index", "middle") and f != roll_mode:
                    rolling_this = False
                if rolling_this:
                    ramp = 1.0 if ROLL_RAMP_S <= 0 else min(1.0, (t - carry_until) / ROLL_RAMP_S)
                    if roll_mode == "extend" and j in ("pip", "dip"):
                        voluntary = -ramp * limit
                    elif roll_mode == "mcp" and j != "mcp":
                        voluntary = 0.0
                    else:
                        voluntary = ramp * limit
                elif f not in grip.roll_fingers and t >= open_at and (f == "thumb" or OPEN_EXTENDS_SUPPORT):
                    voluntary = -limit
                else:
                    voluntary = 0.0
                tau = float(np.clip(tau_id + voluntary, -limit, limit))
                data.ctrl[aid] = tau / limit
                continue
            if f in grip.roll_fingers and t >= carry_until and roll_mode != "hold":
                if roll_mode in ("index", "middle") and f != roll_mode:
                    target = init + GRIP_SQUEEZE_RAD
                    data.ctrl[aid] = target
                    continue
                if not rolling[aid]:
                    _as_motor(model, aid, TORQUE_CAP_NM[j] * torque_scale)
                    rolling[aid] = True
                ramp = 1.0 if ROLL_RAMP_S <= 0 else min(1.0, (t - carry_until) / ROLL_RAMP_S)
                if release_step is not None:
                    data.ctrl[aid] = 0.0
                elif roll_mode == "extend" and j in ("pip", "dip"):
                    data.ctrl[aid] = -ramp
                elif roll_mode == "mcp" and j != "mcp":
                    data.ctrl[aid] = 0.0
                else:
                    data.ctrl[aid] = ramp
                continue
            # No extra curl while the ball is being carried. A position target
            # past the leather, against a ball that yields, walks the ball out
            # of the hand over the delivery. The roll torque starts at carry_until.
            target = init if t < carry_until else init + GRIP_SQUEEZE_RAD * (f == "thumb" or f in grip.roll_fingers)
            if f not in grip.roll_fingers and t >= open_at:
                target = JOINT_LIMITS[j][0] + 0.05 if (f == "thumb" or OPEN_EXTENDS_SUPPORT) else init
            data.ctrl[aid] = target
        # The carry is a placed ball. Ball contacts stay off until the roll,
        # because a welded ball the pads cannot move is an infinite wall.
        # Foot–mound contact stays on for the whole delivery, including the
        # set: only the leather geom is toggled.
        carry = t < carry_until
        model.opt.disableflags &= ~contact_off
        _set_leather(release_step is None and not carry)
        if carry:
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_comVel(model, data)
            R_palm = np.array(data.site_xmat[palm_sid]).reshape(3, 3)
            w_palm, v_palm = _site_velocity(model, data, palm_sid)
            held = data.site_xpos[palm_sid] + R_palm @ carry_local
            # 0: sit on the leather. The pulp travel is how far a node can
            # retract once the roll's contacts are on, not a preload.
            allow = 0.0 if carry_allow_m is None else carry_allow_m
            pos = _lift_off_pads(data, held, pad_ids, belly_ids, allow)
            offset = pos - data.site_xpos[palm_sid]
            data.qpos[bq : bq + 3] = pos
            data.qpos[bq + 3 : bq + 7] = quat_from_mat(R_palm @ carry_rot)
            data.qvel[bv : bv + 3] = v_palm + np.cross(w_palm, offset)
            if carry_spin == "zero":
                data.qvel[bv + 3 : bv + 6] = 0.0
            elif carry_spin == "palm":
                data.qvel[bv + 3 : bv + 6] = (R_palm @ carry_rot).T @ w_palm
            elif carry_spin == "notop":
                # Keep the palm's spin except the topspin/backspin part (+X).
                # The X component is what the upright forearm paints on as topspin.
                w_world = w_palm.copy()
                w_world[0] = 0.0
                data.qvel[bv + 3 : bv + 6] = (R_palm @ carry_rot).T @ w_world
            else:
                raise ValueError(f"carry_spin must be 'palm', 'zero', or 'notop', got {carry_spin!r}")
        if os.environ.get("PITCHLAB_DEBUG_STEP") and 0.0 <= t <= 0.0015:
            ia = int(np.argmax(np.abs(scratch.qacc)))
            itau = int(np.argmax(np.abs(data.qfrc_applied)))
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.dof_jntid[ia]))
            tname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.dof_jntid[itau]))
            gh0 = int(J.v["gh"])
            q_ref, _, a_ref = ref.ball["gh"]
            q = data.qpos[J.q["gh"] : J.q["gh"] + 4]
            err = quat_log(quat_mul(quat_conj(q), q_ref[i]))
            print(
                f"pre t={t:.5f} qacc[{ia}:{jname}]={scratch.qacc[ia]:+.1f} "
                f"tau[{itau}:{tname}]={data.qfrc_applied[itau]:+.1f} "
                f"gh_err={np.linalg.norm(err):.4f} gh_aref={np.linalg.norm(a_ref[i]):.1f} "
                f"gh_tau={np.array2string(data.qfrc_applied[gh0:gh0+3], precision=0)} "
                f"pel={np.array2string(data.qfrc_applied[pv:pv+6], precision=1)}",
                flush=True,
            )
        mujoco.mj_step(model, data)
        mujoco.mj_rnePostConstraint(model, data)

        # Everything below reads the state mj_step started from (time t).
        min_finger = min(min_finger, _finger_joint_min(model, data))
        sh = data.site_xpos[sid["shoulder"]].copy()
        el = data.site_xpos[sid["elbow"]].copy()
        wr = data.site_xpos[sid["wrist"]].copy()
        R_up = _rot(data, bids["upper_trunk"])
        R_pel = _rot(data, bids["pelvis"])
        R_hum = _rot(data, bids["humerus"])
        R_fore = _rot(data, bids["forearm"])
        w_up = _body_omega(model, data, bids["upper_trunk"])
        w_hum = _body_omega(model, data, bids["humerus"])
        d = (el - sh) / np.linalg.norm(el - sh)
        log["t"].append(t)
        log["pelvis_vy"].append(float(data.qvel[pv + 1]))
        log["lead_knee_deg"].append(math.degrees(float(data.qpos[J.q["lead_knee"]])))
        if guided:
            lead_f, trail_f = _support_split(
                root_force,
                data.subtree_com[pelvis_bid],
                data.site_xpos[ankle_sids["lead"]],
                data.site_xpos[ankle_sids["trail"]],
                block,
            )
        else:
            lead_f = _geom_force(model, data, foot_gids["lead"])
            trail_f = _geom_force(model, data, foot_gids["trail"])
        log["grf_lead_y"].append(float(lead_f[1]))
        log["grf_lead_z"].append(float(lead_f[2]))
        log["grf_trail_y"].append(float(trail_f[1]))
        log["grf_trail_z"].append(float(trail_f[2]))
        log["pelvis_w"].append(math.degrees(_body_omega(model, data, bids["pelvis"])[2]))
        log["trunk_w"].append(math.degrees(w_up[2]))
        log["elbow_w"].append(-math.degrees(data.qvel[J.v["elbow"]]))
        log["ir_w"].append(-math.degrees(float((w_hum - w_up) @ d)))
        log["wrist_w"].append(math.degrees(data.qvel[J.v["wrist_flex"]]))
        log["elbow_deg"].append(math.degrees(data.qpos[J.q["elbow"]]))
        log["sep"].append(math.degrees(_yaw(R_pel[:, 1]) - _yaw(R_up[:, 1])))
        log["fer"].append(fleisig_er_deg(R_up[:, 1], R_up[:, 2], sh, el, wr))
        # Fleisig's and OBP's trunk tilt is the hip-center to thorax line, which
        # includes lumbar flexion; the thorax segment's own axis is ~8° less.
        zt = data.site_xpos[sid["thorax_prox"]] - data.xpos[bids["pelvis"]]
        log["fwd"].append(math.degrees(math.atan2(zt[1], zt[2])))
        log["lat"].append(math.degrees(math.atan2(-zt[0], zt[2])))
        # Stodden's torso angle: the shoulder line's heading in the horizontal plane.
        log["rot"].append(math.degrees(math.atan2(R_up[1, 0], R_up[0, 0])))
        f_s, m_s = _interaction(model, data, bids["humerus"], sh)
        f_e, m_e = _interaction(model, data, bids["forearm"], el)
        log["ir_moment"].append(float(m_s @ R_hum[:, 2]))
        log["sh_comp"].append(float(f_s @ R_hum[:, 2]))
        log["varus"].append(float(m_e @ R_fore[:, 1]))
        log["el_comp"].append(float(f_e @ R_fore[:, 2]))
        hand_err.append(float(np.linalg.norm(wr - ref.points["wrist_jc"][i])))
        ball_w = _rot(data, ball_bid) @ data.qvel[bv + 3 : bv + 6]
        log["ball_wx"].append(float(ball_w[0]))
        log["ball_wy"].append(float(ball_w[1]))
        log["ball_wz"].append(float(ball_w[2]))
        log["contacts"].append(float(len(_touching(model, data))))
        if t + 1e-12 >= carry_until and release_step is None and i % 10 == 0:
            probed = _contact_probe(model, data, t, bq, bv, ball_bid)
            for name, force in probed["forces"].items():
                finger_force[name] = max(finger_force.get(name, 0.0), float(np.linalg.norm(force)))
                if name in grip.roll_fingers or name == "thumb":
                    finger_trace.append((t, name, force.copy(), probed["pads"].get(name, np.zeros(3)).copy()))

        if t >= carry_until:
            if _touching(model, data):
                seen = True
                quiet = 0
            elif seen:
                quiet += 1
                if release_step is None and quiet >= quiet_need:
                    # Index into the logged samples. Skipping early steps with
                    # PITCHLAB_T_START must not point this at the full grid.
                    release_step = len(log["t"]) - 1 - quiet_need
                    release_velocity = data.qvel[bv : bv + 3].copy()
                    release_omega = _rot(data, ball_bid) @ data.qvel[bv + 3 : bv + 6]
                    release_pos = data.qpos[bq : bq + 3].copy()
                    # Free flight: fingers swinging through afterward must not re-strike it.
                    # Feet stay in contact through the follow-through.
                    _set_leather(False)
        if not sampled and t + 1e-12 >= release:
            hand_speed = float(np.linalg.norm(_site_velocity(model, data, sid["wrist"])[1]))
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, data.qpos[bq : bq + 3].copy(), bids["hand"])
            for name in (*BALLS, *HINGES):
                width = 3 if name in BALLS else 1
                cols = slice(J.v[name], J.v[name] + width)
                contributions[name] = jac[:, cols] @ data.qvel[cols]
            contributions["pelvis"] = jac[:, pv : pv + 6] @ data.qvel[pv : pv + 6]
            sampled = True
        if probe is not None and t <= release + 0.012 and i % 20 == 0:
            probe.append(_contact_probe(model, data, t, bq, bv, ball_bid))
            probe[-1]["fingers"] = {(f, j): float(data.qpos[q]) for f, j, _, q, _, _ in fingers}
            probe[-1]["forearm"] = (wr - el) / max(float(np.linalg.norm(wr - el)), 1e-9)
            probe[-1]["up"] = np.array([0.0, 0.0, 1.0])
            probe[-1]["palm_v"] = _site_velocity(model, data, palm_sid)[1]
            probe[-1]["held_v"] = probe[-1]["palm_v"] + np.cross(
                _site_velocity(model, data, palm_sid)[0], np.array(data.site_xmat[palm_sid]).reshape(3, 3) @ carry_local
            )
            probe[-1]["joints"] = {
                n: (float(data.qpos[J.q[n]]), float(ref.hinge[n][0][i]), float(data.qvel[J.v[n]]))
                for n in ("elbow", "pronation", "wrist_flex", "wrist_dev")
            }
            probe[-1]["applied"] = {n: float(data.qfrc_applied[J.v[n]]) for n in ("elbow", "pronation", "wrist_flex", "wrist_dev")}
            probe[-1]["ball_err"] = {
                n: quat_log(quat_mul(quat_conj(data.qpos[J.q[n] : J.q[n] + 4]), ref.ball[n][0][i]))
                for n in ("lumbar", "thoracic", "gh")
            }
            probe[-1]["ball_w"] = {
                n: (data.qvel[J.v[n] : J.v[n] + 3].copy(), ref.ball[n][1][i].copy()) for n in ("lumbar", "thoracic", "gh")
            }
        if store_frames and t + 1e-12 >= next_frame:
            frames.append(data.qpos.copy())
            next_frame += frame_dt
        if os.environ.get("PITCHLAB_DEBUG_STEP") and -0.0001 <= t <= 0.15 and i % 200 == 0:
            pz = data.qpos[pq : pq + 3]
            soles = []
            for side in ("lead", "trail"):
                gid = foot_gids[side]
                soles.append(float(data.geom_xpos[gid][2] - model.geom_size[gid][0]))
            ktau = int(np.argmax(np.abs(data.qfrc_applied)))
            gh0 = int(J.v["gh"])
            qh = data.qpos[J.q["lead_hip"] : J.q["lead_hip"] + 4]
            herr = quat_log(quat_mul(quat_conj(qh), plant_q.get("lead_hip", qh)))
            print(
                f"dbg t={t:.4f} pelz={pz[2]:+.3f} vy={data.qvel[pv+1]:+.2f} vz={data.qvel[pv+2]:+.2f} "
                f"knee={math.degrees(float(data.qpos[J.q['lead_knee']])):+.1f} "
                f"ktgt={math.degrees(plant_h.get('lead_knee', float('nan'))):+.1f} "
                f"hipz={plant_watch.get('lead', (0, 0, 0))[0]:+.3f} "
                f"az={plant_watch.get('lead', (0, 0, 0))[1]:+.3f} "
                f"reach={plant_watch.get('lead', (0, 0, 0))[2]:.3f} "
                f"herr={math.degrees(float(np.linalg.norm(herr))):.1f} "
                f"ncon={data.ncon} sole={soles[0]:+.4f},{soles[1]:+.4f} "
                f"Fy={lead_f[1]:+.0f} Fz={lead_f[2]:+.0f} "
                f"IRint={log['ir_moment'][-1]:+.0f} "
                f"gh={np.array2string(data.qfrc_applied[gh0:gh0+3], precision=0)} "
                f"|tau|={float(data.qfrc_applied[ktau]):+.1f}@{ktau}",
                flush=True,
            )
        if os.environ.get("PITCHLAB_DEBUG_UNTIL") and t > float(os.environ["PITCHLAB_DEBUG_UNTIL"]):
            break

    if release_velocity is None:
        release_velocity = data.qvel[bv : bv + 3].copy()
        release_omega = _rot(data, ball_bid) @ data.qvel[bv + 3 : bv + 6]
        release_pos = data.qpos[bq : bq + 3].copy()
    times = np.array(log["t"])
    series = {k: np.array(v) for k, v in log.items()}
    for key in ("ir_moment", "varus", "sh_comp", "el_comp"):
        series[f"{key}_raw"] = series[key]
        series[key] = lowpass(series[key], 1.0 / TIMESTEP_S)
    result = _summarize(grip, motion, body, ref, series, release_velocity, release_omega, release_pos, release_step,
                        min_finger, sq_err, n_err, hand_err, contributions, hand_speed, times)
    result.push_off_ms = release * 1000.0
    result.finger_force_n = finger_force
    release_t = float(times[release_step]) if release_step is not None else float(times[-1])
    result.finger_force_parts = _finger_force_parts(finger_trace, release_t)
    result.ball_id = get_ball(ball).id

    def _signed_peak(xs: list[float]) -> float:
        return float(max(xs, key=abs)) if xs else 0.0

    above = {}
    for name in YIELD_HINGES:
        n = len(distal_raw[name])
        n_over = sum(abs(a - b) > 1e-3 for a, b in zip(distal_raw[name], distal_ceiling[name]))
        above[name] = n_over / n if n else 0.0
    deviation = {}
    for name, rows in whip_err.items():
        if not rows:
            deviation[name] = {"rms": 0.0, "max": 0.0, "at_release": 0.0}
            continue
        err = np.degrees([q - q_ref_n for _, q, q_ref_n in rows])
        deviation[name] = {
            "rms": float(np.sqrt(np.mean(err * err))),
            "max": float(np.max(np.abs(err))),
            "at_release": float(err[-1]),
        }
    k_rel = int(release_step) if release_step is not None else len(times) - 1
    k_fc = int(np.argmin(np.abs(times)))
    bw = max(body.mass_kg * 9.81, 1e-6)
    post = times >= -1e-12

    def _gpeak(key: str, how: str) -> float:
        seg = series[key][post] / bw
        if len(seg) == 0:
            return 0.0
        if how == "min":
            return float(np.min(seg))
        return float(np.max(seg))

    grf = {
        side: {
            "fy_min": _gpeak(ykey, "min"),
            "fy_max": _gpeak(ykey, "max"),
            "fz_min": _gpeak(zkey, "min"),
            "fz_max": _gpeak(zkey, "max"),
        }
        for side, ykey, zkey in (
            ("lead", "grf_lead_y", "grf_lead_z"),
            ("trail", "grf_trail_y", "grf_trail_z"),
        )
    }
    # Set is the hold before the knee lift (max knee is near −0.8 s).
    # Landing is the first 80 ms after the lead foot is down.
    set_mask = times < -1.05
    plant_mask = (times >= 0.005) & (times <= 0.080)

    def _median_bw(key: str, mask: np.ndarray) -> float:
        seg = series[key][mask]
        if len(seg) == 0:
            return 0.0
        return float(np.median(seg) / bw)
    result.distal_drive = {
        "mode": "forward" if distal_forward else "tracking",
        "body_mode": body_mode,
        "grf_source": "inverse-dynamics root force, lever rule" if guided else "foot-mound contact",
        "tracking_rms_by_joint_deg": {
            n: math.degrees(math.sqrt(track_sq[n] / max(track_n[n], 1))) for n in track_sq
        },
        "tracking_max_by_joint_deg": {n: (math.degrees(v), ms * 1000.0) for n, (v, ms) in track_max.items()},
        "raw_peak_nm": {n: _signed_peak(distal_raw[n]) for n in YIELD_HINGES},
        "applied_peak_nm": {n: _signed_peak(distal_applied[n]) for n in YIELD_HINGES},
        "isometric_ceiling_nm": {n: _signed_peak(distal_ceiling[n]) for n in YIELD_HINGES},
        "above_isometric_fraction": above,
        "wrist_flex_vs_ref_deg": deviation.get("wrist_flex", {}).get("at_release", 0.0),
        "wrist_deviation_deg": deviation,
        "pelvis_vy_release_m_s": float(series["pelvis_vy"][k_rel]),
        "lead_knee_fc_deg": float(series["lead_knee_deg"][k_fc]),
        "lead_knee_rel_deg": float(series["lead_knee_deg"][k_rel]),
        "grf_bw": grf,
        "stance_fz_set_bw": _median_bw("grf_trail_z", set_mask),
        "lead_fz_plant_bw": _median_bw("grf_lead_z", plant_mask),
        "lead_fy_plant_bw": _median_bw("grf_lead_y", plant_mask),
        "trail_fy_push_bw": _median_bw("grf_trail_y", (times >= -0.20) & (times < -0.01)),
        "wrist_emg_peak": dict(WRIST_EMG_PEAK),
    }
    if store_frames:
        return result, model, frames, ref
    return result


def _reference_kinematics(ref: Reference) -> dict[str, np.ndarray]:
    """The same metrics computed on the measured joint centers."""
    p = ref.points
    n = len(ref.t)
    fer = np.zeros(n)
    for i in range(n):
        R = _thorax_from_points(p, i)
        fer[i] = fleisig_er_deg(R[:, 1], R[:, 2], p["shoulder_jc"][i], p["elbow_jc"][i], p["wrist_jc"][i])
    return {"fer": fer}


def _summarize(grip, motion, body, ref, s, v, w, pos, release_step, min_finger, sq_err, n_err, hand_err,
               contributions, hand_speed, times) -> DeliveryResult:
    release = ref.release
    speed = float(np.linalg.norm(v))
    rpm = float(np.linalg.norm(w) * 60.0 / (2.0 * math.pi))
    v_rel, w_rel = np.array([-v[0], v[1], v[2]]), np.array([w[0], -w[1], -w[2]])
    tilt, eff = tilt_and_efficiency(w_rel, v_rel)
    cycle = (times >= -0.02) & (times <= release + 0.02)

    def peak(key: str) -> tuple[float, float]:
        vals = np.where(cycle, s[key], -np.inf)
        k = int(np.argmax(vals))
        return float(vals[k]), float(times[k] * 1000.0)

    def load_peak(key: str, fc: float | None = None) -> tuple[float, float]:
        series = s[key] if fc is None else lowpass(s[f"{key}_raw"], 1.0 / TIMESTEP_S, fc)
        vals = np.where((times >= 0.0) & (times <= release + AFTER_RELEASE_S), series, -np.inf)
        k = int(np.argmax(vals))
        return float(vals[k]), float(times[k] * 1000.0)

    cock = (times > 0.0) & (times < release)
    k_mer = int(np.argmax(np.where(cock, s["fer"], -np.inf)))
    k_rel = int(np.argmin(np.abs(times - release)))
    k_fc = int(np.argmin(np.abs(times)))
    ref_fer = _reference_kinematics(ref)["fer"][: len(times)]
    ref_mer = float(np.max(np.where(cock, ref_fer, -np.inf)))

    # Reference peaks from the trajectories themselves (no dynamics).
    def ref_peak(values: np.ndarray) -> tuple[float, float]:
        vals = np.where(cycle, values[: len(times)], -np.inf)
        k = int(np.argmax(vals))
        return float(vals[k]), float(times[k] * 1000.0)

    pel_q = ref.pelvis["quat"]
    qm = quat_mul

    def world_w(quat: np.ndarray, w_body: np.ndarray) -> np.ndarray:
        qw = qm(qm(quat, np.concatenate([np.zeros((len(w_body), 1)), w_body], 1)), quat_conj(quat))
        return qw[:, 1:]

    q_mid = qm(pel_q, ref.ball["lumbar"][0])
    q_up = qm(q_mid, ref.ball["thoracic"][0])
    w_pel = world_w(pel_q, ref.pelvis["w"])
    w_mid_rel = world_w(q_mid, ref.ball["lumbar"][1])
    w_up = w_pel + w_mid_rel + world_w(q_up, ref.ball["thoracic"][1])
    reference_peaks = {
        "pelvis": ref_peak(np.degrees(w_pel[:, 2])),
        "trunk": ref_peak(np.degrees(w_up[:, 2])),
        "elbow extension": ref_peak(-np.degrees(ref.hinge["elbow"][1])),
        "wrist flexion": ref_peak(np.degrees(ref.hinge["wrist_flex"][1])),
    }
    hand_rel = ref.points["wrist_jc"][k_rel] - ref.points["shoulder_jc"][k_rel]
    arm_slot = math.degrees(math.atan2(hand_rel[2], abs(hand_rel[0])))
    rubber_front = 0.08
    return DeliveryResult(
        pitch_id=motion.id,
        grip=grip.id,
        speed_m_s=speed,
        speed_mph=speed * 2.2369362920544,
        radar_mph=motion.radar_m_s * 2.2369362920544,
        spin_rpm=rpm,
        spin_efficiency=eff,
        tilt_deg=tilt,
        clock=_clock(tilt),
        velocity=np.array(v),
        omega=np.array(w),
        separated=release_step is not None,
        release_ms=(float(times[release_step]) if release_step is not None else float(times[-1])) * 1000.0,
        obp_release_ms=release * 1000.0,
        release_height_m=float(pos[2]),
        release_side_m=float(pos[0]),
        extension_m=float(pos[1] - rubber_front),
        min_finger_rad=min_finger,
        peaks={
            "pelvis": peak("pelvis_w"),
            "trunk": peak("trunk_w"),
            "elbow extension": peak("elbow_w"),
            "internal rotation": peak("ir_w"),
            "wrist flexion": peak("wrist_w"),
        },
        reference_peaks=reference_peaks,
        mer_deg=float(s["fer"][k_mer]),
        mer_ms=float(times[k_mer] * 1000.0),
        reference_mer_deg=ref_mer,
        separation_at_fc_deg=float(s["sep"][k_fc]),
        max_separation_deg=float(np.max(np.where(cycle, s["sep"], -np.inf))),
        trunk_forward_at_release_deg=float(s["fwd"][k_rel]),
        trunk_lateral_at_release_deg=float(s["lat"][k_rel]),
        trunk_rotation_at_release_deg=float(s["rot"][k_rel]),
        elbow_at_release_deg=float(s["elbow_deg"][k_rel]),
        arm_slot_deg=arm_slot,
        tracking_rms_deg=math.degrees(math.sqrt(sq_err / max(n_err, 1))),
        hand_error_mm=1000.0 * float(np.sqrt(np.mean(np.square(np.array(hand_err)[cycle])))),
        loads={
            "shoulder IR moment": load_peak("ir_moment"),
            "elbow varus moment": load_peak("varus"),
            "shoulder compression": load_peak("sh_comp"),
            "elbow compression": load_peak("el_comp"),
        },
        fleisig_loads={
            "shoulder IR moment": load_peak("ir_moment", FLEISIG_CUTOFF_HZ),
            "elbow varus moment": load_peak("varus", FLEISIG_CUTOFF_HZ),
            "shoulder compression": load_peak("sh_comp", FLEISIG_CUTOFF_HZ),
        },
        obp_loads=dict(motion.reference_loads),
        hand_speed_at_release_m_s=hand_speed,
        obp_hand_speed_at_release_m_s=float(
            np.linalg.norm(np.gradient(ref.points["wrist_jc"], ref.t, axis=0)[k_rel])
        ),
        contributions=contributions,
        series=s,
        height_m=body.height_m,
        mass_kg=body.mass_kg,
        pronation_offset_deg=ref.pronation_offset_deg,
    )


def _distal_line(r: DeliveryResult) -> str:
    d = r.distal_drive
    if not d or "raw_peak_nm" not in d:
        return ""
    parts = [
        "  distal raw/applied N·m: "
        + "  ".join(f"{n} {d['raw_peak_nm'][n]:.1f}/{d['applied_peak_nm'][n]:.1f}" for n in d["raw_peak_nm"])
    ]
    if "pelvis_vy_release_m_s" in d:
        parts.append(
            f"  pelvis origin vy at release {d['pelvis_vy_release_m_s']:+.2f} m/s  "
            f"lead knee at plant {d['lead_knee_fc_deg']:.1f}°  at release {d['lead_knee_rel_deg']:.1f}°"
        )
    if "grf_bw" in d:
        g = d["grf_bw"]
        parts.append(
            "  foot GRF after contact, plate on body (BW): "
            + "  ".join(
                f"{side} Fy {g[side]['fy_min']:+.2f} to {g[side]['fy_max']:+.2f}  Fz {g[side]['fz_min']:+.2f} to {g[side]['fz_max']:+.2f}"
                for side in ("lead", "trail")
            )
        )
    if "wrist_deviation_deg" in d:
        w = d["wrist_deviation_deg"]
        parts.append(
            "  whip angle minus reference (rms / max deg): "
            + "  ".join(f"{n} {w[n]['rms']:.1f}/{w[n]['max']:.1f}" for n in ("wrist_flex", "wrist_dev", "pronation") if n in w)
        )
    return "\n".join(parts)


def format_result(r: DeliveryResult) -> str:
    v = r.velocity
    lines = [
        f"{r.pitch_id} ({r.height_m:.2f} m, {r.mass_kg:.0f} kg, reference radar {r.radar_mph:.1f} mph)  grip {r.grip}  ball {r.ball_id}",
        f"  ball {r.speed_m_s:.1f} m/s ({r.speed_mph:.1f} mph)  {r.spin_rpm:.0f} rpm  axis {r.clock}  eff {r.spin_efficiency:.2f}  "
        f"vel ({v[0]:+.1f}, {v[1]:+.1f}, {v[2]:+.1f})",
        f"  leaves fingers at {r.release_ms:.1f} ms (push-off {r.push_off_ms:.1f}, reference release {r.obp_release_ms:.1f} ms)  "
        f"launch {r.launch_deg:+.1f}° vertical, {r.launch_horizontal_deg:+.1f}° toward glove side  "
        f"height {r.release_height_m:.2f} m  side {r.release_side_m:+.2f} m  extension {r.extension_m:.2f} m",
        "  peaks (integrated / reference): "
        + "  ".join(
            f"{k} {val:.0f}"
            + (f"/{r.reference_peaks[k][0]:.0f}" if k in r.reference_peaks else "")
            + f"°/s @ {ms:.0f} ms"
            for k, (val, ms) in r.peaks.items()
        ),
        f"  MER (Fleisig) {r.mer_deg:.0f}° @ {r.mer_ms:.0f} ms (reference {r.reference_mer_deg:.0f}°)  "
        f"separation FC {r.separation_at_fc_deg:.0f}° max {r.max_separation_deg:.0f}°  "
        f"trunk at release fwd {r.trunk_forward_at_release_deg:.0f}° lat {r.trunk_lateral_at_release_deg:.0f}° "
        f"rot {r.trunk_rotation_at_release_deg:.0f}°  elbow {r.elbow_at_release_deg:.0f}°  arm slot {r.arm_slot_deg:.0f}°",
        f"  wrist speed at release {r.hand_speed_at_release_m_s:.1f} m/s (reference {r.obp_hand_speed_at_release_m_s:.1f})  "
        f"tracking rms {r.tracking_rms_deg:.2f}°  wrist position rms {r.hand_error_mm:.0f} mm  "
        f"pronation offset {r.pronation_offset_deg:.0f}°",
        "  loads (model / source, same pitch): "
        + "  ".join(
            f"{k} {val:.0f} @ {ms:.0f} ms"
            + (f" / {r.obp_loads[k][0]:.0f} @ {r.obp_loads[k][1]:.0f} ms" if k in r.obp_loads else "")
            for k, (val, ms) in r.loads.items()
        ),
        "  loads at Fleisig's 13.4 Hz: "
        + "  ".join(f"{k} {val:.0f} @ {ms:.0f} ms" for k, (val, ms) in r.fleisig_loads.items()),
        _distal_line(r),
        "  ball-point velocity by joint at reference release (x, y, z m/s): "
        + "  ".join(
            f"{k} ({c[0]:+.1f}, {c[1]:+.1f}, {c[2]:+.1f})"
            for k, c in sorted(r.contributions.items(), key=lambda kv: -float(np.linalg.norm(kv[1])))
            if float(np.linalg.norm(c)) > 0.8
        ),
    ]
    return "\n".join(lines)


def validate(motions: list[Motion], grip: str = "four-seam") -> list[DeliveryResult]:
    """Same model and settings on several motions. Nothing is tuned per motion."""
    return [simulate_delivery(grip, m) for m in motions]


_DEFAULT: list[Motion] = []


def default_motion() -> Motion:
    """The single-pitcher delivery (`literature.literature_motion`), built once."""
    if not _DEFAULT:
        from pitchlab.literature import literature_motion

        _DEFAULT.append(literature_motion())
    return _DEFAULT[0]


# ---------------------------------------------------------------- video


def render_delivery(
    model: mujoco.MjModel,
    frames: list[np.ndarray],
    path: str,
    ref: Reference,
    result: DeliveryResult,
    fps: int = 30,
    frame_dt: float = 0.002,
) -> None:
    """Side (third-base side) and behind, full body, in slow motion."""
    import subprocess

    if not frames:
        raise RuntimeError("no frames")
    width, height = 640, 720
    renderer = mujoco.Renderer(model, height, width)
    data = mujoco.MjData(model)
    t0 = float(ref.t[0])

    def pelvis_at(t: float) -> np.ndarray:
        data.qpos[:] = frames[min(len(frames) - 1, int(round((t - t0) / frame_dt)))]
        mujoco.mj_forward(model, data)
        return _site_pos(model, data, "pelvis")

    look = (pelvis_at(t0) + pelvis_at(0.0) + pelvis_at(float(ref.release))) / 3.0
    look[2] = 0.95
    views = (
        ("side  ·  home is to the left", 0.0, -4.0),
        ("behind the pitcher", -90.0, -4.0),
    )
    events = {
        "lead knee high": float(ref.events.get("max_knee_time", t0)),
        "foot contact": 0.0,
        "max external rotation": result.mer_ms / 1000.0,
        "ball leaves fingers": result.release_ms / 1000.0,
    }
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width * len(views)}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdin is not None
    slow = int(round(1.0 / (fps * frame_dt)))
    try:
        for index, frame in enumerate(frames):
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            t = t0 + index * frame_dt
            marks = [name for name, at in events.items() if abs(t - at) < 0.006]
            panels = []
            for title, azimuth, elevation in views:
                cam = mujoco.MjvCamera()
                cam.distance = 5.2
                cam.azimuth = azimuth
                cam.elevation = elevation
                cam.lookat[:] = look
                renderer.update_scene(data, camera=cam)
                lines = [title, f"{t * 1000.0:6.0f} ms from foot contact  (1/{slow} speed)"] + marks
                panels.append(_stamp(renderer.render(), lines, accent=bool(marks)))
            proc.stdin.write(np.ascontiguousarray(np.concatenate(panels, axis=1)).tobytes())
    finally:
        proc.stdin.close()
        proc.wait(timeout=300)
        renderer.close()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited {proc.returncode}")
