"""Five-finger release with a free ball.

The arm joints follow published angular velocities kinematically: the arm is
much heavier than the ball, and this window does not try to reproduce the
whole delivery. Finger joints are position servos with a torque limit, so the
pulp can comply. The ball has a free joint and raised seams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from pitchlab.constants import (
    ACCEL_MS,
    ARM_SLOT_DEG,
    BALL_RADIUS_M,
    CONTACT_MS,
    CONTACT_SOLIMP,
    CONTACT_SOLREF,
    DIP_EXT_RAD,
    DIP_FLEX_RAD,
    DIP_TORQUE_NM,
    ELBOW_EXTENSION_DEG_S,
    ELBOW_FLEXION_AT_RELEASE_DEG,
    FINGER_FLEX_DEG_S,
    FOLLOW_MS,
    FOREARM_M,
    HUMERUS_ABDUCTION_DEG,
    HUMERUS_M,
    IR_SIGMA_S,
    MCP_EXT_RAD,
    MCP_FLEX_RAD,
    MCP_TORQUE_NM,
    PAD_RADIUS_M,
    PIP_EXT_RAD,
    PULP_DAMPING,
    PULP_NODE_RADIUS_M,
    PULP_PROTRUSION_M,
    PULP_STIFFNESS_N_PER_M,
    PULP_TRAVEL_M,
    PIP_FLEX_RAD,
    PIP_TORQUE_NM,
    PRONATION_FOLLOW_CENTER_MS,
    PRONATION_FOLLOW_DEG_S,
    PRONATION_FOLLOW_WINDOW_MS,
    PROXIMAL_SPEED_MS,
    SEAM_BUMP_RADIUS_M,
    SEAM_BACKING_STRIDE,
    SEAM_COUNT,
    SHOULDER_IR_DEG_S,
    THUMB_OFF_MS,
    TIMESTEP_S,
    WRIST_FLEX_DEG_S,
    smoothstep_delta,
)
from pitchlab.balls import contact_mu
from pitchlab.grips import GRIPS, Grip
from pitchlab.seams import seam_samples

FINGERS = ("index", "middle", "ring", "pinky", "thumb")
# Segment lengths in meters, adult male defaults from the lab hand profile.
SEGMENTS_M = {
    "index": (0.040, 0.024, 0.020),
    "middle": (0.044, 0.026, 0.022),
    "ring": (0.041, 0.025, 0.020),
    "pinky": (0.032, 0.020, 0.016),
    "thumb": (0.034, 0.032, 0.026),
}
# Knuckle position in the hand frame. Palm faces +Z, fingers point +X.
# Knuckles sit on the distal edge of the palm so a 7.4 cm ball fits in the hand.
ROOTS = {
    "index": (0.062, -0.018, 0.012),
    "middle": (0.066, 0.000, 0.013),
    "ring": (0.062, 0.017, 0.012),
    "pinky": (0.054, 0.033, 0.010),
    "thumb": (0.020, -0.036, 0.000),
}
# Opposition, so the thumb pad can sit on the ball instead of pointing away from it.
THUMB_EULER = "-20 10 70"
# XML joint ranges are degrees. Actuator ctrlrange is NOT converted by the
# compiler's angle setting, so those limits are written in radians.
JOINT_LIMITS = {
    "mcp": (-MCP_EXT_RAD, MCP_FLEX_RAD),
    "pip": (-PIP_EXT_RAD, PIP_FLEX_RAD),
    "dip": (-DIP_EXT_RAD, DIP_FLEX_RAD),
}
_MOUNT: dict[str, object] = {}


@dataclass
class ReleaseResult:
    grip: str
    speed_m_s: float
    speed_mph: float
    spin_rpm: float
    spin_efficiency: float
    tilt_deg: float
    omega: np.ndarray
    velocity: np.ndarray
    wrist_speed_m_s: float
    peak_contacts: int
    contact_ms: float
    separated: bool
    # Smallest finger-joint angle during the clip. Anatomical stops are negative
    # by only a few degrees; a large negative value is hyperextension.
    min_finger_rad: float
    # Palm +Z in the world, dotted with +Y (home), sampled at the arm's release.
    palm_forward: float
    # Forearm direction's vertical component at release. Steeply negative means
    # the forearm is still pointing at the ground.
    forearm_up: float
    arm_angle_deg: float
    # How far the wrist is in front of the head along +Y at release.
    ahead_of_head_m: float
    # Last time each finger touched the ball, in ms relative to the arm release.
    finger_last_ms: dict[str, float]


def _clock(deg: float) -> str:
    d = deg % 360
    minutes = int(round((d / 360) * 12 * 60)) % (12 * 60)
    hours = minutes // 60
    mins = minutes % 60
    if hours == 0:
        hours = 12
    return f"{hours}:{mins:02d}"


def tilt_and_efficiency(omega: np.ndarray, velocity: np.ndarray) -> tuple[float, float]:
    vm = float(np.linalg.norm(velocity))
    om = float(np.linalg.norm(omega))
    if vm < 1e-8 or om < 1e-8:
        return 0.0, 0.0
    vhat = velocity / vm
    force = np.cross(omega, vhat)
    force = force - vhat * float(np.dot(force, vhat))
    up = np.array([0.0, 0.0, 1.0])
    up = up - vhat * float(np.dot(up, vhat))
    if np.linalg.norm(up) < 1e-8:
        up = np.array([0.0, 0.0, 1.0])
    up = up / np.linalg.norm(up)
    right = np.array([-1.0, 0.0, 0.0])
    right = right - vhat * float(np.dot(right, vhat))
    right = right - up * float(np.dot(right, up))
    rn = float(np.linalg.norm(right))
    if rn < 1e-8:
        right = np.array([-1.0, 0.0, 0.0])
    else:
        right = right / rn
    tilt = math.degrees(math.atan2(float(np.dot(force, right)), float(np.dot(force, up)))) % 360
    parallel = min(1.0, abs(float(np.dot(omega, velocity))) / (om * vm))
    efficiency = math.sqrt(max(0.0, 1.0 - parallel * parallel))
    return tilt, efficiency


def _smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _smooth_rate(u: float, duration: float) -> float:
    if u <= 0.0 or u >= 1.0 or duration <= 0:
        return 0.0
    return (6 * u * (1 - u)) / duration


def _range_deg(joint: str) -> str:
    lo, hi = JOINT_LIMITS[joint]
    return f"{math.degrees(lo):.2f} {math.degrees(hi):.2f}"


def _pulp_centers() -> tuple[np.ndarray, ...]:
    """Node centers on the palmar side of the 9 mm envelope.

    Palm is +Z, the fingertip points +X. Two rings sit on a sphere of radius
    PAD_RADIUS − node radius, so the free surface is the old sphere. Nodes
    with x < −2.5 mm are the proximal/dorsal side and are left off. Spacing
    is about 3.0–3.5 mm, wide enough for a 0.80 mm seam bump to meet a flank.
    """
    radius = PAD_RADIUS_M + PULP_PROTRUSION_M - PULP_NODE_RADIUS_M
    centers = [np.array([0.0, 0.0, radius])]
    for theta_deg, n, phase in ((28.0, 7, 0.0), (56.0, 11, 0.5)):
        theta = math.radians(theta_deg)
        ring = radius * math.sin(theta)
        z = radius * math.cos(theta)
        for k in range(n):
            phi = 2.0 * math.pi * (k + phase) / n
            center = np.array([ring * math.cos(phi), ring * math.sin(phi), z])
            if center[0] > -0.0025:
                centers.append(center)
    return tuple(centers)


PULP_TIP_CENTERS = _pulp_centers()
# Second contact of a four-seam finger: on the distal pad, proximal of the tip.
# 10 mm back and 8 mm to the palmar side is as far as this finger still lies on
# the ball; further proximal the leather gap opens again. The length is
# assumed. It is the span that can reach both legs of the narrow horseshoe
# (Bourke's closest approach is 8.6 mm). The wide mouth of the U is farther
# than this pad can reach.
# Center of the proximal patch. The patch itself is wider (see _belly_centers):
# the two fingers do not put the second leg at the same local coordinate.
BELLY_LOCAL_M = np.array([-0.008, -0.002, 0.010])


def _belly_centers() -> tuple[np.ndarray, ...]:
    """Nodes on the distal pad, proximal of the tip, including the thumb side.

    The second horseshoe leg of the adopted fingertip grip does not land
    here. A disc the width of a measured fingertip (Jung et al. 2015, index
    DIP 18.8 mm) can cover both legs only when the pad is centered on the
    open leather, and that pitch missed the release window. These nodes stay
    inside an 8 mm half-width.
    """
    z = 0.010 - PULP_NODE_RADIUS_M
    centers = []
    for x in (-0.005, -0.009, -0.012):
        for y in (-0.008, -0.003, 0.002):
            centers.append(np.array([x, y, z]))
    return tuple(centers)


PULP_BELLY_CENTERS = _belly_centers()
PULP_CENTERS = PULP_TIP_CENTERS + PULP_BELLY_CENTERS
# Only the fingers that stay on the ball through the fastball roll. A node on
# the thumb was catching a seam before the roll and turning the launch.
PULP_FINGERS = ("index", "middle")


def _finger_of_pad(name: str) -> str | None:
    if name == "palm_pad":
        return "palm"
    for finger in FINGERS:
        head = f"{finger}_pad"
        if name == head or name.startswith(head + "_"):
            return finger
    return None


# Node-seam solver. CONTACT_SOLIMP's width is 1 mm, larger than the seam, so
# the backing keeps that soft pair (the adopted pitch). Nodes use a 0.08 mm
# width so the measured 0.80 mm height can push a flank. Friction stays at
# the Yamaguchi with-seam value. The width is assumed.
SEAM_SOLIMP = "0.99 0.999 0.00008 0.5 2"
SEAM_SOLREF = "0.002 1"


def _pair(pad: str, seam: int, mu: float, solref: str, solimp: str, margin: float) -> str:
    return (
        f'<pair geom1="{pad}" geom2="seam_{seam}" condim="4" '
        f'friction="{mu:.4f} 0.02 0.001" solref="{solref}" '
        f'solimp="{solimp}" margin="{margin:.5f}"/>'
    )


def seam_pair_xml(seam_mu: float, pulp: bool = True) -> str:
    lines = []
    backing = [f"{finger}_pad" for finger in FINGERS]
    backing.append("palm_pad")
    for i in range(SEAM_COUNT):
        if i % SEAM_BACKING_STRIDE == 0:
            for pad in backing:
                lines.append(_pair(pad, i, seam_mu, CONTACT_SOLREF, CONTACT_SOLIMP, 0.0004))
        if pulp:
            for finger in PULP_FINGERS:
                for n in range(len(PULP_CENTERS)):
                    lines.append(_pair(f"{finger}_pad_{n:02d}", i, seam_mu, SEAM_SOLREF, SEAM_SOLIMP, 0.00005))
    return "".join(lines)


def _pulp_xml(name: str, belly_leather: bool = False) -> str:
    """Small radial slides outside a smooth backing sphere.

    The backing (geom ``{name}_pad``) is the old 9 mm sphere and touches only
    leather, so the no-seam grip is unchanged. These nodes touch only the
    seam. They stand 1 mm proud and retract flush when a bump pushes them;
    a neighbor that is beside the bump stays out and the flank hooks it.
    """
    mass = 0.004 / len(PULP_CENTERS)
    parts = [f'<site name="{name}_pad" pos="0 0 0" size="0.004"/>']
    if belly_leather:
        # Proximal pad under the tip sphere. Only the four-seam grip uses it:
        # on the changeup it shoves the ball off the throw line. mass 0 so the
        # node inertials still carry the pulp mass.
        parts.append(
            f'<geom class="pad" name="{name}_pad_belly" type="capsule" '
            f'fromto="-0.012 -0.008 0.006 -0.005 0.003 0.006" size="0.0045" mass="0"/>'
        )
    for i, center in enumerate(PULP_CENTERS):
        axis = center / np.linalg.norm(center)
        parts.append(
            f'<body name="{name}_pulp_{i:02d}" pos="{center[0]:.5f} {center[1]:.5f} {center[2]:.5f}">'
            f'<joint name="{name}_pulp_{i:02d}" type="slide" axis="{axis[0]:.5f} {axis[1]:.5f} {axis[2]:.5f}" '
            f'range="{-PULP_TRAVEL_M:.5f} 0" limited="true" ref="0" '
            f'stiffness="{PULP_STIFFNESS_N_PER_M:.1f}" damping="{PULP_DAMPING:.3f}" armature="2e-5" '
            f'solreflimit="0.001 1" solimplimit="0.99 0.999 0.00005 0.5 2"/>'
            f'<inertial pos="0 0 0" mass="{mass:.6f}" diaginertia="1e-8 1e-8 1e-8"/>'
            f'<geom class="pad" name="{name}_pad_{i:02d}" size="{PULP_NODE_RADIUS_M:.5f}" density="0" '
            f'contype="4" conaffinity="8"/>'
            f"</body>"
        )
    return "".join(parts)


def _finger_xml(name: str, abduction_deg: float, pulp: bool = True, belly_leather: bool = False) -> str:
    proximal, middle, distal = SEGMENTS_M[name]
    x, y, z = ROOTS[name]
    euler = THUMB_EULER if name == "thumb" else f"0 0 {abduction_deg:.2f}"
    bone = 'density="0" contype="0" conaffinity="0" rgba="0.93 0.78 0.68 1"'
    use_pulp = pulp and name in PULP_FINGERS
    return f"""
        <body name="{name}_root" pos="{x:.4f} {y:.4f} {z:.4f}" euler="{euler}">
          <joint name="{name}_mcp" type="hinge" axis="0 -1 0" damping="0.03" range="{_range_deg("mcp")}" limited="true"/>
          <inertial pos="{proximal * 0.5:.4f} 0 0" mass="0.012" diaginertia="2e-6 2e-6 4e-7"/>
          <geom type="capsule" fromto="0 0 0 {proximal:.4f} 0 0" size="0.006" {bone}/>
          <body pos="{proximal:.4f} 0 0">
            <joint name="{name}_pip" type="hinge" axis="0 -1 0" damping="0.02" range="{_range_deg("pip")}" limited="true"/>
            <inertial pos="{middle * 0.5:.4f} 0 0" mass="0.008" diaginertia="1e-6 1e-6 2e-7"/>
            <geom type="capsule" fromto="0 0 0 {middle:.4f} 0 0" size="0.0055" {bone}/>
            <body pos="{middle:.4f} 0 0">
              <joint name="{name}_dip" type="hinge" axis="0 -1 0" damping="0.015" range="{_range_deg("dip")}" limited="true"/>
              <inertial pos="{distal * 0.5:.4f} 0 0" mass="0.005" diaginertia="6e-7 6e-7 1e-7"/>
              <geom type="capsule" fromto="0 0 0 {distal:.4f} 0 0" size="0.005" {bone}/>
              <body pos="{distal:.4f} 0 0">
                {(_pulp_xml(name, belly_leather) if use_pulp else f'<site name="{name}_pad" pos="0 0 0" size="0.004"/>')}
                <geom class="pad" name="{name}_pad" size="{PAD_RADIUS_M}"/>
              </body>
            </body>
          </body>
        </body>"""


def _actuator(name: str, torque: float, joint: str) -> str:
    lo, hi = JOINT_LIMITS[joint]
    return (
        f'<position name="{name}_act" joint="{name}" kp="18" kv="0.35" '
        f'forcerange="-{torque} {torque}" ctrlrange="{lo:.4f} {hi:.4f}"/>'
    )


def build_model(grip: Grip, friction: float | None = None, pulp: bool = True, ball=None) -> mujoco.MjModel:
    spec, friction, seam_mu = contact_mu(ball, friction)
    seam_center = spec.radius_m - (SEAM_BUMP_RADIUS_M - spec.seam_height_m)
    seams = []
    for i, p in enumerate(seam_samples(SEAM_COUNT)):
        pos = p * seam_center
        seams.append(
            f'<geom class="seam" name="seam_{i}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}"/>'
        )
    fingers = "\n".join(
        _finger_xml(name, grip.abduction_deg[name], pulp, belly_leather=grip.id == "four-seam") for name in FINGERS
    )
    acts = []
    for name in FINGERS:
        acts.append(_actuator(f"{name}_mcp", MCP_TORQUE_NM, "mcp"))
        acts.append(_actuator(f"{name}_pip", PIP_TORQUE_NM, "pip"))
        acts.append(_actuator(f"{name}_dip", DIP_TORQUE_NM, "dip"))
    # MuJoCo's default pair friction is the pairwise maximum, so a seam geom
    # never wins against a pad at the leather coefficient. Seams are excluded
    # from automatic contact and paired explicitly at the with-seam coefficient.
    fr = f"{friction:.4f} 0.02 0.001"
    mount = _ensure_mount()
    pairs = seam_pair_xml(seam_mu, pulp)
    xml = f"""
<mujoco model="pitch-release">
  <option timestep="{TIMESTEP_S}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" iterations="60"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.62 0.62 0.58" diffuse="0.65 0.64 0.58"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.70 0.84" rgb2="0.93 0.90 0.82" width="512" height="256"/>
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
    <light pos="0.6 -1.5 3.2" dir="-0.2 0.45 -1"/>
    <geom name="ground" type="plane" size="6 12 0.1" pos="0 4 0" rgba="0.46 0.36 0.24 1" contype="0" conaffinity="0"/>
    <geom name="lane" type="box" size="0.045 7 0.004" pos="0 7 0.004" rgba="0.86 0.82 0.68 1" contype="0" conaffinity="0"/>
    <body name="ball">
      <freejoint name="ball_free"/>
      <geom name="leather" type="sphere" size="{spec.radius_m}" mass="{spec.mass_kg}"
            friction="{fr}" condim="4" solref="{CONTACT_SOLREF}" solimp="{CONTACT_SOLIMP}"
            contype="1" conaffinity="2" rgba="0.95 0.93 0.88 1"/>
      <site name="ball_center" pos="0 0 0"/>
      {''.join(seams)}
    </body>
    <body name="proximal">
      <joint name="proximal" type="slide" axis="0 1 0" damping="0"/>
      <inertial pos="0 0 1.1" mass="4" diaginertia="0.05 0.05 0.05"/>
      <geom name="torso" type="capsule" fromto="0 0 0.90 0 0 1.42" size="0.12" density="0" contype="0" conaffinity="0" rgba="0.15 0.22 0.38 1"/>
      <geom name="head" type="sphere" pos="0 -0.02 1.66" size="0.11" density="0" contype="0" conaffinity="0" rgba="0.93 0.78 0.66 1"/>
      <site name="head" pos="0 -0.02 1.66" size="0.01"/>
      <body name="shoulder" pos="-0.18 0.04 1.46">
        <site name="shoulder" pos="0 0 0" size="0.01"/>
        <body name="humerus_align" quat="{mount["humerus_quat"]}">
          <joint name="internal_rotation" type="hinge" axis="1 0 0" damping="0.05"/>
          <inertial pos="0.12 0 0" mass="2.2" diaginertia="0.02 0.02 0.004"/>
          <geom type="capsule" fromto="0 0 0 {HUMERUS_M:.3f} 0 0" size="0.022" density="0" contype="0" conaffinity="0" rgba="0.86 0.62 0.48 1"/>
          <body name="elbow_anchor" pos="{HUMERUS_M} 0 0">
            <site name="elbow" pos="0 0 0" size="0.01"/>
            <joint name="elbow" type="hinge" axis="{mount["elbow_axis"]}" damping="0.05"/>
            <inertial pos="{FOREARM_M * 0.4:.3f} 0 0" mass="1.15" diaginertia="0.008 0.008 0.001"/>
            <geom type="capsule" fromto="0 0 0 {FOREARM_M:.3f} 0 0" size="0.018" density="0" contype="0" conaffinity="0" rgba="0.90 0.70 0.56 1"/>
            <body name="wrist_anchor" pos="{FOREARM_M} 0 0">
              <joint name="pronation" type="hinge" axis="1 0 0" damping="0.01"/>
              <inertial pos="0 0 0" mass="0.15" diaginertia="0.0004 0.0004 0.0002"/>
              <body name="wrist">
                <joint name="wrist_flex" type="hinge" axis="0 1 0" damping="0.01"/>
                <inertial pos="0 0 0" mass="0.08" diaginertia="0.0002 0.0002 0.0001"/>
                <site name="wrist" pos="0 0 0" size="0.008"/>
                <body name="hand" quat="{mount["hand_quat"]}">
                  <inertial pos="0.04 0 0" mass="0.45" diaginertia="0.001 0.001 0.0004"/>
                  <geom type="box" pos="0.034 0 0.004" size="0.038 0.034 0.010" density="0" contype="0" conaffinity="0" rgba="0.93 0.76 0.64 1"/>
                  <geom class="pad" name="palm_pad" pos="0.030 0.000 0.016" size="0.013"/>
                  <site name="palm" pos="0.030 0.000 0.016" size="0.006"/>
                  {fingers}
                </body>
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


def _jid(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(name)
    return int(jid)


def _set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, q: float, qd: float | None = None) -> None:
    jid = _jid(model, name)
    data.qpos[int(model.jnt_qposadr[jid])] = q
    if qd is not None:
        data.qvel[int(model.jnt_dofadr[jid])] = qd


def _release_time() -> float:
    """Arm pose defined as release: internal rotation is fastest, elbow is at 22°."""
    return ACCEL_MS / 1000.0


def _burst(t: float, center: float, window: float, peak_deg_s: float) -> tuple[float, float]:
    """Smoothstep angle and rate of a burst centered at `center`."""
    half = window / 2.0
    u = (t - (center - half)) / window
    delta = smoothstep_delta(peak_deg_s, window)
    return delta * _smooth(u), delta * _smooth_rate(u, window)


def _ir_profile(t: float, sign: float) -> tuple[float, float]:
    """Brief internal-rotation spike centered on release. Angle is 0 at release."""
    peak = math.radians(SHOULDER_IR_DEG_S)
    sigma = IR_SIGMA_S
    z = (t - _release_time()) / sigma
    rate = peak * math.exp(-0.5 * z * z)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    angle = peak * sigma * math.sqrt(2.0 * math.pi) * (cdf - 0.5)
    return sign * angle, sign * rate


def _elbow_profile(t: float) -> tuple[float, float]:
    """Elbow is still extending at release. The published peak is slightly before it.

    The smoothstep's fastest point is the middle of the burst. The burst ends
    just after release, so the angle is 22° at release and the rate is not zero.
    """
    end = _release_time() + 0.008
    peak = math.radians(ELBOW_EXTENSION_DEG_S)
    # Peak smoothstep slope is 1.5 / duration, so this angle change hits that peak.
    delta = peak * end / 1.5
    u_rel = _release_time() / end
    # Fraction of the extension still left at release.
    remain = 1.0 - _smooth(u_rel)
    finish = math.radians(ELBOW_FLEXION_AT_RELEASE_DEG) - delta * remain
    start = finish + delta
    u = min(1.0, max(0.0, t / end))
    angle = start + (finish - start) * _smooth(u)
    rate = (finish - start) * _smooth_rate(u, end) if t < end else 0.0
    return angle, rate


def _arm(t: float, grip: Grip) -> dict[str, tuple[float, float]]:
    """Joint position and velocity. Release is the zero of internal rotation."""
    sign = float(_MOUNT["ir_sign"])
    internal = _ir_profile(t, sign)
    elbow = _elbow_profile(t)
    # Barrentine: peak pronation is after release, and it is pronation for every pitch.
    pron_q, pron_qd = _burst(
        t,
        _release_time() + PRONATION_FOLLOW_CENTER_MS / 1000.0,
        PRONATION_FOLLOW_WINDOW_MS / 1000.0,
        PRONATION_FOLLOW_DEG_S,
    )
    # Peak wrist flexion is +3 ms. Half of a short burst still overlaps the roll.
    wrist_q, wrist_qd = _burst(
        t,
        _release_time() + 0.003,
        0.016,
        WRIST_FLEX_DEG_S,
    )
    return {
        "proximal": (PROXIMAL_SPEED_MS * t, PROXIMAL_SPEED_MS),
        "internal_rotation": internal,
        "elbow": elbow,
        "pronation": (math.radians(grip.pronation_start_deg) + pron_q, pron_qd),
        "wrist_flex": (math.radians(grip.wrist_start_deg) - wrist_q, -wrist_qd),
    }


def _clamp_joint(joint: str, angle: float) -> float:
    lo, hi = JOINT_LIMITS[joint]
    return min(hi, max(lo, angle))


def _finger_target(t: float, grip: Grip, finger: str, joint: str, initial: float) -> float:
    """Thumb, ring, and pinky leave first. Index and middle (or the grip's roll fingers) stay."""
    t_rel = _release_time()
    lo, _hi = JOINT_LIMITS[joint]
    opened = max(lo + 0.02, 0.10 if joint == "mcp" else 0.05)
    if finger not in grip.roll_fingers:
        # Stevenson: thumb off ~6 ms before release. The other early fingers share that clock.
        open_at = t_rel - THUMB_OFF_MS / 1000.0
        if t <= open_at:
            return _clamp_joint(joint, initial)
        u = (t - open_at) / (THUMB_OFF_MS / 1000.0)
        return _clamp_joint(joint, initial + (opened - initial) * _smooth(u))

    contact = CONTACT_MS / 1000.0
    u = (t - (t_rel - contact)) / contact
    delta = smoothstep_delta(FINGER_FLEX_DEG_S, contact)
    if joint == "dip":
        delta *= 0.67
    held = initial + delta * _smooth(min(1.0, max(0.0, u)))
    # After the ball should be gone, ease toward a mild flexion instead of staying clamped.
    if t > t_rel + 0.004:
        relax = 0.40 if joint == "mcp" else 0.25
        v = min(1.0, (t - t_rel - 0.004) / 0.020)
        held = held + (relax - held) * _smooth(v)
    return _clamp_joint(joint, held)


def _quat_str(q: np.ndarray) -> str:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    return f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"


def _mul_quat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


def _align_x_quat(direction: np.ndarray, twist: float) -> np.ndarray:
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    axis = np.array([1.0, 0.0, 0.0])
    cross = np.cross(axis, d)
    dot = float(np.dot(axis, d))
    q_align = np.array([1.0 + dot, cross[0], cross[1], cross[2]])
    if 1.0 + dot < 1e-8:
        q_align = np.array([0.0, 0.0, 1.0, 0.0])
    q_align = q_align / np.linalg.norm(q_align)
    half = twist / 2.0
    q_twist = np.array([math.cos(half), math.sin(half), 0.0, 0.0])
    return _mul_quat(q_align, q_twist)


def _light_model(humerus_quat: str, elbow_axis: str, hand_quat: str) -> mujoco.MjModel:
    xml = f"""
<mujoco>
  <worldbody>
    <body name="proximal">
      <joint name="proximal" type="slide" axis="0 1 0"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <site name="head" pos="0 -0.02 1.66"/>
      <body name="shoulder" pos="-0.18 0.04 1.46">
        <site name="shoulder" pos="0 0 0"/>
        <body name="humerus_align" quat="{humerus_quat}">
          <joint name="internal_rotation" type="hinge" axis="1 0 0"/>
          <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
          <body name="elbow_anchor" pos="{HUMERUS_M} 0 0">
            <site name="elbow" pos="0 0 0"/>
            <joint name="elbow" type="hinge" axis="{elbow_axis}"/>
            <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
            <body name="wrist_anchor" pos="{FOREARM_M} 0 0">
              <joint name="pronation" type="hinge" axis="1 0 0"/>
              <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001"/>
              <body name="wrist">
                <joint name="wrist_flex" type="hinge" axis="0 1 0"/>
                <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
                <site name="wrist" pos="0 0 0"/>
                <body name="hand" quat="{hand_quat}">
                  <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
                  <site name="palm" pos="0.030 0 0.016"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)


def _pose(model: mujoco.MjModel, ir: float, elbow: float, wrist: float, pronation: float) -> mujoco.MjData:
    data = mujoco.MjData(model)
    _set_joint(model, data, "internal_rotation", ir, 0.0)
    _set_joint(model, data, "elbow", elbow, 0.0)
    _set_joint(model, data, "wrist_flex", wrist, 0.0)
    _set_joint(model, data, "pronation", pronation, 0.0)
    mujoco.mj_forward(model, data)
    return data


def _ensure_mount() -> dict[str, object]:
    """Pick the humerus twist whose release pose faces home near the RHP arm slot."""
    if _MOUNT:
        return _MOUNT
    elev = math.radians(HUMERUS_ABDUCTION_DEG - 90.0)
    identity = "1 0 0 0"
    best: tuple[float, dict[str, object]] | None = None
    ff = GRIPS["four-seam"]
    wrist_q = math.radians(ff.wrist_start_deg)
    pron_q = math.radians(ff.pronation_start_deg)
    elbow_rel, _ = _elbow_profile(_release_time())
    elbow_start, _ = _elbow_profile(0.0)
    for forward_deg in (12.0, 20.0, 28.0):
        forward = math.radians(forward_deg)
        direction = np.array(
            [
                -math.cos(elev) * math.cos(forward),
                math.cos(elev) * math.sin(forward),
                math.sin(elev),
            ]
        )
        direction = direction / np.linalg.norm(direction)
        for twist_deg in range(0, 360, 15):
            quat = _align_x_quat(direction, math.radians(twist_deg))
            quat_s = _quat_str(quat)
            for axis in ("0 1 0", "0 -1 0"):
                model = _light_model(quat_s, axis, identity)
                check = _pose(model, 0.0, 0.0, 0.0, 0.0)
                bone = _site_pos(model, check, "elbow") - _site_pos(model, check, "shoulder")
                bone = bone / np.linalg.norm(bone)
                if float(np.dot(bone, direction)) < 0.98:
                    raise RuntimeError(f"humerus did not align: {bone} vs {direction}")
                for sign in (1.0, -1.0):
                    ir0, _ = _ir_profile(0.0, sign)
                    _, ir_rate = _ir_profile(_release_time(), sign)
                    _, elbow_rate = _elbow_profile(_release_time())
                    t_mid = _release_time() * 0.65
                    ir_mid, ir_rate_mid = _ir_profile(t_mid, sign)
                    elbow_mid, elbow_rate_mid = _elbow_profile(t_mid)
                    start = _pose(model, ir0, elbow_start, wrist_q, pron_q)
                    release = _pose(model, 0.0, elbow_rel, wrist_q, pron_q)
                    mid = _pose(model, ir_mid, elbow_mid, wrist_q, pron_q)
                    _set_joint(model, release, "proximal", 0.0, PROXIMAL_SPEED_MS)
                    _set_joint(model, release, "internal_rotation", 0.0, ir_rate)
                    _set_joint(model, release, "elbow", elbow_rel, elbow_rate)
                    mujoco.mj_forward(model, release)
                    vel = _site_lin_vel(model, release, "wrist")
                    speed = float(np.linalg.norm(vel))
                    _set_joint(model, mid, "proximal", PROXIMAL_SPEED_MS * t_mid, PROXIMAL_SPEED_MS)
                    _set_joint(model, mid, "internal_rotation", ir_mid, ir_rate_mid)
                    _set_joint(model, mid, "elbow", elbow_mid, elbow_rate_mid)
                    mujoco.mj_forward(model, mid)
                    vel_mid = _site_lin_vel(model, mid, "wrist")
                    speed_mid = float(np.linalg.norm(vel_mid))
                    forearm0 = _site_pos(model, start, "wrist") - _site_pos(model, start, "elbow")
                    forearm1 = _site_pos(model, release, "wrist") - _site_pos(model, release, "elbow")
                    forearm0 = forearm0 / np.linalg.norm(forearm0)
                    forearm1 = forearm1 / np.linalg.norm(forearm1)
                    wrist = _site_pos(model, release, "wrist")
                    head = _site_pos(model, release, "head")
                    slot = math.degrees(math.atan2(forearm1[2], math.hypot(forearm1[0], forearm1[1])))
                    ahead = float(wrist[1] - head[1])
                    forward = float(vel[1]) / speed if speed > 1e-6 else -1.0
                    forward_mid = float(vel_mid[1]) / speed_mid if speed_mid > 1e-6 else -1.0
                    # Release velocity toward home is not enough: the whole arc has to
                    # be moving that way, or the hand swipes sideways and throws the ball out.
                    score = (
                        8.0 * forward
                        + 5.0 * forward_mid
                        + 0.2 * float(vel[1])
                        + 4.0 * float(forearm1[1])
                        + 1.2 * float(forearm1[2])
                        - 3.5 * abs(float(forearm1[0]))
                        - 4.0 * abs(float(vel[0])) / max(speed, 1e-6)
                        - 4.0 * abs(float(vel_mid[0])) / max(speed_mid, 1e-6)
                        - 4.0 * max(0.0, -float(vel[2])) / max(speed, 1e-6)
                        - 4.0 * max(0.0, -float(vel_mid[1])) / max(speed_mid, 1e-6)
                        - 3.0 * max(0.0, -float(vel_mid[2])) / max(speed_mid, 1e-6)
                        + 2.0 * ahead
                        + 0.8 * float(forearm0[2])
                        + 0.4 * float(wrist[1] - _site_pos(model, start, "wrist")[1])
                        - 0.02 * abs(slot - ARM_SLOT_DEG)
                        - 5.0 * max(0.0, -0.15 - float(forearm0[2]))
                    )
                    chosen = {
                        "humerus_quat": quat_s,
                        "elbow_axis": axis,
                        "ir_sign": sign,
                        "forward_deg": forward_deg,
                        "twist_deg": twist_deg,
                        "score": score,
                        "slot": slot,
                    }
                    if best is None or score > best[0]:
                        best = (score, chosen)
    assert best is not None
    chosen = best[1]
    # Orient the hand so that, at the four-seam release joints, the palm faces home
    # and the fingers point up. Other grips differ by their own wrist and pronation.
    model = _light_model(str(chosen["humerus_quat"]), str(chosen["elbow_axis"]), identity)
    t_rel = _release_time()
    wrist_extra, _ = _burst(t_rel, t_rel + 0.003, 0.016, WRIST_FLEX_DEG_S)
    pron_extra, _ = _burst(
        t_rel,
        t_rel + PRONATION_FOLLOW_CENTER_MS / 1000.0,
        PRONATION_FOLLOW_WINDOW_MS / 1000.0,
        PRONATION_FOLLOW_DEG_S,
    )
    release = _pose(
        model,
        0.0,
        elbow_rel,
        math.radians(ff.wrist_start_deg) - wrist_extra,
        math.radians(ff.pronation_start_deg) + pron_extra,
    )
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "palm")
    current = np.array(release.site_xmat[sid], dtype=float).reshape(3, 3)
    # Right-handed. +X fingers point up, +Z palm points toward home, +Y is the pinky side.
    desired = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    # Measured against the compiled body-quat convention used by this MuJoCo build.
    mount = desired.T @ current
    quat = np.zeros(4)
    # mju_mat2Quat reads a column-major matrix.
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(mount.flatten(order="F")))
    chosen["hand_quat"] = _quat_str(quat)
    _MOUNT.update(chosen)
    return _MOUNT


def _initial_finger(grip: Grip, finger: str, joint: str) -> float:
    mcp = math.radians(grip.mcp_deg[finger])
    if joint == "mcp":
        return mcp
    if joint == "pip":
        return 0.85 * mcp
    return 0.55 * mcp


def _apply_arm(model: mujoco.MjModel, data: mujoco.MjData, t: float, grip: Grip) -> None:
    for name, (q, qd) in _arm(t, grip).items():
        _set_joint(model, data, name, q, qd)


def _site_lin_vel(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, sid, res, 0)
    return res[3:].copy()


def _site_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return data.site_xpos[sid].copy()


def wrist_state(grip: Grip | None = None) -> tuple[np.ndarray, float]:
    """Wrist linear velocity at mid-release, before contact is involved."""
    grip = grip or GRIPS["four-seam"]
    model = build_model(grip)
    data = mujoco.MjData(model)
    _apply_arm(model, data, _release_time(), grip)
    for finger in FINGERS:
        for joint in ("mcp", "pip", "dip"):
            _set_joint(model, data, f"{finger}_{joint}", _initial_finger(grip, finger, joint), 0.0)
    mujoco.mj_forward(model, data)
    vel = _site_lin_vel(model, data, "wrist")
    return vel, float(np.linalg.norm(vel))


def _ball_quat(grip: Grip) -> np.ndarray:
    rx, ry, rz = [math.radians(v) for v in grip.ball_euler_deg]
    def q_axis(angle: float, axis: np.ndarray) -> np.ndarray:
        s = math.sin(angle / 2)
        return np.array([math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])

    def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    q = mul(mul(q_axis(rx, np.array([1.0, 0, 0])), q_axis(ry, np.array([0, 1.0, 0]))), q_axis(rz, np.array([0, 0, 1.0])))
    return q / np.linalg.norm(q)


def _place_ball(model: mujoco.MjModel, data: mujoco.MjData, grip: Grip) -> None:
    mujoco.mj_forward(model, data)
    palm = _site_pos(model, data, "palm")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "palm")
    normal = np.array(data.site_xmat[sid], dtype=float).reshape(3, 3)[:, 2]
    normal = normal / np.linalg.norm(normal)
    if grip.id == "changeup":
        pad_names = ["ring_pad", "middle_pad"]
    elif grip.id == "slider":
        pad_names = ["index_pad", "middle_pad"]
    else:
        pad_names = ["index_pad", "middle_pad"]
    a = _site_pos(model, data, pad_names[0])
    b = _site_pos(model, data, pad_names[1])
    mid = 0.5 * (a + b)
    # Sit the ball on the palm, then slide it toward the fingertips so they can roll it.
    center = palm + normal * (BALL_RADIUS_M + 0.011 - grip.depth_m)
    toward_fingers = mid - center
    toward_fingers = toward_fingers - normal * float(np.dot(toward_fingers, normal))
    length = float(np.linalg.norm(toward_fingers))
    if length > 1e-6:
        toward_fingers = toward_fingers / length
    across = np.cross(normal, toward_fingers)
    across_n = float(np.linalg.norm(across))
    across = across / across_n if across_n > 1e-8 else np.array([1.0, 0.0, 0.0])
    center = center + toward_fingers * 0.008 + across * grip.side_m
    adr = int(model.jnt_qposadr[_jid(model, "ball_free")])
    data.qpos[adr : adr + 3] = center
    data.qpos[adr + 3 : adr + 7] = _ball_quat(grip)
    vel_adr = int(model.jnt_dofadr[_jid(model, "ball_free")])
    data.qvel[vel_adr : vel_adr + 3] = _site_lin_vel(model, data, "palm")
    data.qvel[vel_adr + 3 : vel_adr + 6] = 0.0


def _finger_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    count = 0
    for i in range(data.ncon):
        pair = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
            for g in (data.contact[i].geom1, data.contact[i].geom2)
        ]
        pad = next((name for name in pair if _finger_of_pad(name)), "")
        other = next((name for name in pair if name != pad), "")
        if pad and (other == "leather" or other.startswith("seam_")):
            count += 1
    return count


def _touching(model: mujoco.MjModel, data: mujoco.MjData) -> set[str]:
    names: set[str] = set()
    for i in range(data.ncon):
        pair = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
            for g in (data.contact[i].geom1, data.contact[i].geom2)
        ]
        pad = next((name for name in pair if _finger_of_pad(name)), "")
        other = next((name for name in pair if name != pad), "")
        finger = _finger_of_pad(pad) if pad else None
        if finger and (other == "leather" or other.startswith("seam_")):
            names.add(finger)
    return names


def _finger_joint_min(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    worst = 1e9
    for finger in FINGERS:
        for joint in ("mcp", "pip", "dip"):
            jid = _jid(model, f"{finger}_{joint}")
            worst = min(worst, float(data.qpos[int(model.jnt_qposadr[jid])]))
    return worst


def _pose_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float, float, float, float]:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "palm")
    rotation = np.array(data.site_xmat[sid], dtype=float).reshape(3, 3)
    palm_forward = float(rotation[:, 2] @ np.array([0.0, 1.0, 0.0]))
    wrist = _site_pos(model, data, "wrist")
    elbow = _site_pos(model, data, "elbow")
    forearm = wrist - elbow
    forearm_up = float(forearm[2] / np.linalg.norm(forearm))
    delta = _site_pos(model, data, "ball_center") - _site_pos(model, data, "shoulder")
    arm_angle = math.degrees(math.atan2(delta[2], math.hypot(float(delta[0]), float(delta[1]))))
    ahead = float(wrist[1] - _site_pos(model, data, "head")[1])
    wrist_speed = float(np.linalg.norm(_site_lin_vel(model, data, "wrist")))
    return palm_forward, forearm_up, arm_angle, ahead, wrist_speed


def _site_ang_vel(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, sid, res, 0)
    return res[:3].copy()


def _capture_carry(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    """Remember where the ball sits in the palm frame so the arm can carry it."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "palm")
    rotation = np.array(data.site_xmat[sid], dtype=float).reshape(3, 3)
    adr = int(model.jnt_qposadr[_jid(model, "ball_free")])
    palm = data.site_xpos[sid]
    return {
        "local": rotation.T @ (data.qpos[adr : adr + 3] - palm),
        "ball_quat": data.qpos[adr + 3 : adr + 7].copy(),
        "palm_quat": _quat_from_mat(rotation),
    }


def _quat_from_mat(rotation: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation.flatten(order="F")))
    return quat


def _carry_ball(model: mujoco.MjModel, data: mujoco.MjData, carry: dict[str, np.ndarray]) -> None:
    """Keep the ball in the hand until the roll. The release itself is not scripted."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "palm")
    rotation = np.array(data.site_xmat[sid], dtype=float).reshape(3, 3)
    adr = int(model.jnt_qposadr[_jid(model, "ball_free")])
    data.qpos[adr : adr + 3] = data.site_xpos[sid] + rotation @ carry["local"]
    # Keep the seam orientation chosen for the grip. Copying the arm's angular
    # velocity made the ball inherit shoulder rotation instead of finger spin.
    data.qpos[adr + 3 : adr + 7] = carry["ball_quat"]
    vel_adr = int(model.jnt_dofadr[_jid(model, "ball_free")])
    data.qvel[vel_adr : vel_adr + 3] = _site_lin_vel(model, data, "palm")
    data.qvel[vel_adr + 3 : vel_adr + 6] = 0.0


def _set_ctrl(model: mujoco.MjModel, data: mujoco.MjData, t: float, grip: Grip) -> None:
    for finger in FINGERS:
        for joint in ("mcp", "pip", "dip"):
            initial = _initial_finger(grip, finger, joint)
            target = _finger_target(t, grip, finger, joint, initial)
            act = f"{finger}_{joint}_act"
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
            data.ctrl[aid] = target


def simulate(
    grip: Grip | str,
    friction: float | None = None,
    contact_ms: float | None = None,
    ir_deg_s: float | None = None,
    store_frames: bool = False,
    frame_dt: float = 0.0005,
    ball=None,
) -> ReleaseResult | tuple[ReleaseResult, mujoco.MjModel, list[np.ndarray]]:
    """Run one release. Optional overrides are for sensitivity sweeps.

    `ball` is "npb" (default, the unified Mizuno ball) or "mlb". `friction`
    overrides the leather coefficient and scales the seam with it.
    """
    global CONTACT_MS, SHOULDER_IR_DEG_S
    if isinstance(grip, str):
        grip = GRIPS[grip]
    saved_contact = CONTACT_MS
    saved_ir = SHOULDER_IR_DEG_S
    if contact_ms is not None:
        CONTACT_MS = contact_ms
    if ir_deg_s is not None:
        SHOULDER_IR_DEG_S = ir_deg_s
    try:
        return _simulate_grip(grip, friction, store_frames, frame_dt, ball)
    finally:
        CONTACT_MS = saved_contact
        SHOULDER_IR_DEG_S = saved_ir


def _simulate_grip(
    grip: Grip,
    friction: float | None,
    store_frames: bool,
    frame_dt: float,
    ball=None,
) -> ReleaseResult | tuple[ReleaseResult, mujoco.MjModel, list[np.ndarray]]:
    model = build_model(grip, friction, ball=ball)
    data = mujoco.MjData(model)
    _apply_arm(model, data, 0.0, grip)
    for finger in FINGERS:
        for joint in ("mcp", "pip", "dip"):
            _set_joint(model, data, f"{finger}_{joint}", _initial_finger(grip, finger, joint), 0.0)
    _set_ctrl(model, data, 0.0, grip)
    mujoco.mj_forward(model, data)
    _place_ball(model, data, grip)
    mujoco.mj_forward(model, data)
    carry = _capture_carry(model, data)
    # The simplified arm yanks the ball sideways before the hand is moving toward
    # home. Carry it in the palm until the Matsuo roll, then let contact finish the throw.
    carry_until = _release_time() - CONTACT_MS / 1000.0

    duration = (ACCEL_MS + FOLLOW_MS) / 1000.0
    steps = int(duration / TIMESTEP_S)
    frames: list[np.ndarray] = []
    next_frame = 0.0
    peak = 0
    contact_time = 0.0
    seen = False
    release_step = None
    quiet = 0
    quiet_need = int(0.002 / TIMESTEP_S)
    vel_adr = int(model.jnt_dofadr[_jid(model, "ball_free")])
    release_velocity: np.ndarray | None = None
    release_omega: np.ndarray | None = None
    min_finger = 1e9
    sampled = False
    palm_forward = 0.0
    forearm_up = 0.0
    arm_angle = 0.0
    ahead = 0.0
    wrist_speed = 0.0
    last_touch: dict[str, float | None] = {name: None for name in FINGERS}

    for i in range(steps):
        t = i * TIMESTEP_S
        _apply_arm(model, data, t, grip)
        _set_ctrl(model, data, t, grip)
        if t < carry_until:
            _carry_ball(model, data, carry)
        mujoco.mj_step(model, data)
        min_finger = min(min_finger, _finger_joint_min(model, data))
        touching = _touching(model, data)
        for name in touching:
            if name in last_touch:
                last_touch[name] = t
        ncon = len(touching)
        peak = max(peak, ncon)
        # Contact during the kinematic carry is the grip, not the release.
        if t >= carry_until:
            if ncon:
                seen = True
                contact_time += TIMESTEP_S
                quiet = 0
            elif seen:
                quiet += 1
                if release_step is None and quiet >= quiet_need:
                    release_step = i
                    release_velocity = data.qvel[vel_adr : vel_adr + 3].copy()
                    release_omega = data.qvel[vel_adr + 3 : vel_adr + 6].copy()
        if not sampled and t + 1e-12 >= _release_time():
            palm_forward, forearm_up, arm_angle, ahead, wrist_speed = _pose_metrics(model, data)
            sampled = True
        if store_frames and t + 1e-12 >= next_frame:
            frames.append(data.qpos.copy())
            next_frame += frame_dt

    if release_velocity is None or release_omega is None:
        release_velocity = data.qvel[vel_adr : vel_adr + 3].copy()
        release_omega = data.qvel[vel_adr + 3 : vel_adr + 6].copy()
    velocity = release_velocity
    omega = release_omega
    speed = float(np.linalg.norm(velocity))
    rpm = float(np.linalg.norm(omega) * 60 / (2 * math.pi))
    tilt, efficiency = tilt_and_efficiency(omega, velocity)
    finger_last_ms = {
        name: -1.0e9 if last_touch[name] is None else (float(last_touch[name]) - _release_time()) * 1000.0
        for name in FINGERS
    }
    result = ReleaseResult(
        grip=grip.id,
        speed_m_s=speed,
        speed_mph=speed * 2.2369362920544,
        spin_rpm=rpm,
        spin_efficiency=efficiency,
        tilt_deg=tilt,
        omega=omega,
        velocity=velocity,
        wrist_speed_m_s=wrist_speed,
        peak_contacts=peak,
        contact_ms=contact_time * 1000,
        separated=release_step is not None,
        min_finger_rad=min_finger,
        palm_forward=palm_forward,
        forearm_up=forearm_up,
        arm_angle_deg=arm_angle,
        ahead_of_head_m=ahead,
        finger_last_ms=finger_last_ms,
    )
    if store_frames:
        return result, model, frames
    return result


def _stamp(image: np.ndarray, lines: list[str], accent: bool = False) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    fill = (255, 214, 96) if accent else (244, 240, 230)
    y = 14
    for line in lines:
        draw.text((17, y + 1), line, font=font, fill=(12, 12, 12))
        draw.text((16, y), line, font=font, fill=fill)
        y += 28
    return np.asarray(canvas)


def render_video(
    model: mujoco.MjModel,
    frames: list[np.ndarray],
    path: str,
    fps: int = 30,
    frame_dt: float = 0.0005,
) -> None:
    """Side view and behind-the-pitcher view, fixed cameras, release labeled."""
    import subprocess

    if not frames:
        raise RuntimeError("no frames")
    width, height = 640, 480
    renderer = mujoco.Renderer(model, height, width)
    data = mujoco.MjData(model)
    release_index = min(len(frames) - 1, max(0, int(round(_release_time() / frame_dt))))
    data.qpos[:] = frames[release_index]
    mujoco.mj_forward(model, data)
    look = _site_pos(model, data, "wrist")
    # Azimuth 0 puts the camera on +X (pitcher's left), so the frame is the YZ plane.
    # Azimuth -90 puts it on -Y, behind the pitcher, looking toward home.
    # Azimuth 180 is the arm side: +Y (home) moves to the right of the frame.
    # Azimuth -90 is behind the pitcher: +Y moves away, toward the horizon.
    views = (
        ("side view  ·  home is to the right", 180.0, -4.0),
        ("behind the pitcher  ·  home is away", -90.0, 14.0),
    )
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width * 2}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "1",
            path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdin is not None
    try:
        for index, frame in enumerate(frames):
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            panels = []
            at_release = abs(index - release_index) <= 3
            milliseconds = index * frame_dt * 1000.0
            for title, azimuth, elevation in views:
                cam = mujoco.MjvCamera()
                cam.distance = 2.15
                cam.azimuth = azimuth
                cam.elevation = elevation
                cam.lookat[:] = look
                renderer.update_scene(data, camera=cam)
                lines = [title, f"{milliseconds:5.1f} ms"]
                if at_release:
                    lines.append("release")
                panels.append(_stamp(renderer.render(), lines, accent=at_release))
            combined = np.concatenate(panels, axis=1)
            proc.stdin.write(np.ascontiguousarray(combined).tobytes())
    finally:
        proc.stdin.close()
        proc.wait(timeout=180)
        renderer.close()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited {proc.returncode}")


def clock_of(result: ReleaseResult) -> str:
    return _clock(result.tilt_deg)
