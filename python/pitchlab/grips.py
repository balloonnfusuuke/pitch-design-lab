"""Geometric grips. Rates and force limits live in constants and do not change here.

Pronation at release in Barrentine et al. (1998) is about +24° for the fastball
and changeup and about +4° for the curveball. The angles below are not those
measurements. They are hand-vs-ball poses, including the slider offset, chosen
so the contact geometry is a fastball, a curve, a gyro slider, or a changeup.
No paper reports side_m or these MCP angles.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Grip:
    id: str
    name: str
    # Wrist extension at the start of the roll, degrees. Positive is cocked back.
    wrist_start_deg: float
    # Forearm pronation at the start, degrees. The rate sign is pronation_sign.
    pronation_start_deg: float
    pronation_sign: float
    # Initial MCP flexion, degrees, before the shared roll.
    mcp_deg: dict[str, float]
    # Extra abduction, degrees. Positive spreads the finger toward the pinky.
    abduction_deg: dict[str, float]
    last_contact: str
    # Fingers that stay on the ball through the roll. The others open with the
    # thumb, about 6 ms earlier. Four-seam and curve keep index and middle
    # (Kinoshita / Matsuo). The slider keeps the index. The changeup keeps
    # middle and ring.
    roll_fingers: tuple[str, ...]
    # Ball orientation in the hand frame, extrinsic XYZ degrees, so seams meet different pads.
    ball_euler_deg: tuple[float, float, float]
    # Shift of the ball toward the fingers, meters. A choke, not a spin gain.
    depth_m: float
    # Shift across the finger line, meters. Positive moves the ball toward the index side.
    side_m: float


def _abd(**kwargs: float) -> dict[str, float]:
    base = {"index": -6.0, "middle": 0.0, "ring": 7.0, "pinky": 15.0, "thumb": 0.0}
    base.update(kwargs)
    return base


GRIPS: dict[str, Grip] = {
    "four-seam": Grip(
        id="four-seam",
        name="フォーシーム",
        wrist_start_deg=18.0,
        pronation_start_deg=5.0,
        pronation_sign=1.0,
        mcp_deg={"index": 55.0, "middle": 58.0, "ring": 28.0, "pinky": 16.0, "thumb": 40.0},
        abduction_deg=_abd(),
        last_contact="middle",
        roll_fingers=("index", "middle"),
        # Unused by the full-body delivery. That path places the seams with
        # fourseam.four_seam_rotation. Kept so the hand-only model still has a pose.
        ball_euler_deg=(0.0, 8.0, 0.0),
        depth_m=0.0,
        side_m=0.0,
    ),
    "curve": Grip(
        id="curve",
        name="カーブ",
        # Already flexed and more supinated: fingers start farther over the top.
        wrist_start_deg=-12.0,
        # Strongly supinated: the hand is "on top" of the ball. Pose, not a spin gain.
        pronation_start_deg=-70.0,
        pronation_sign=-1.0,
        mcp_deg={"index": 48.0, "middle": 55.0, "ring": 22.0, "pinky": 8.0, "thumb": 34.0},
        abduction_deg=_abd(index=-2.0, middle=2.0),
        last_contact="middle",
        roll_fingers=("index", "middle"),
        ball_euler_deg=(35.0, 20.0, 15.0),
        depth_m=0.0,
        side_m=0.0,
    ),
    "slider": Grip(
        id="slider",
        name="スライダー",
        wrist_start_deg=4.0,
        pronation_start_deg=-55.0,
        pronation_sign=-1.0,
        mcp_deg={"index": 42.0, "middle": 36.0, "ring": 14.0, "pinky": 6.0, "thumb": 40.0},
        # Index sits off to the glove side so the force is not through the center.
        abduction_deg=_abd(index=-28.0, middle=-4.0),
        last_contact="index",
        roll_fingers=("index",),
        ball_euler_deg=(10.0, -25.0, 20.0),
        depth_m=0.0,
        side_m=-0.017,
    ),
    "changeup": Grip(
        id="changeup",
        name="チェンジアップ",
        wrist_start_deg=6.0,
        pronation_start_deg=35.0,
        pronation_sign=1.0,
        # Ball is choked: ring and pinky wrap it, index is barely on it.
        mcp_deg={"index": 12.0, "middle": 28.0, "ring": 46.0, "pinky": 40.0, "thumb": 48.0},
        abduction_deg=_abd(index=4.0, ring=4.0, pinky=10.0),
        last_contact="ring",
        roll_fingers=("middle", "ring"),
        ball_euler_deg=(0.0, -15.0, 40.0),
        depth_m=0.006,
        side_m=0.0,
    ),
}

GRIP_ORDER = ("four-seam", "curve", "slider", "changeup")
