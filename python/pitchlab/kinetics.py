"""Published pitching kinematics and kinetics, for comparison only.

Nothing here drives the simulation; `fullbody` follows OBP motion capture.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Published:
    mean: float
    sd: float
    unit: str
    source: str


# Fleisig, "The biomechanics of baseball pitching" (UAB dissertation), Table 2,
# elite adult means. Velocities are peaks between foot contact and release.
# Kinetics: Fleisig et al., Am J Sports Med 23:233 (1995).
FLEISIG = {
    "pelvis": Published(640.0, 98.0, "deg/s", "Fleisig Table 2, pelvis angular velocity"),
    "trunk": Published(1233.0, 86.0, "deg/s", "Fleisig Table 2, upper torso angular velocity"),
    "elbow extension": Published(2394.0, 312.0, "deg/s", "Fleisig 1995, elbow extension velocity"),
    "internal rotation": Published(7511.0, 1140.0, "deg/s", "Fleisig Table 2, shoulder internal rotation velocity"),
    "mer": Published(179.0, 8.0, "deg", "Fleisig Table 2, maximum external rotation (Fleisig's definition)"),
    "trunk forward at release": Published(37.0, 6.0, "deg", "Fleisig Table 2, 53 ± 6° from horizontal"),
    "trunk lateral at release": Published(31.0, 9.0, "deg", "Fleisig Table 2, 121 ± 9° with 90° vertical"),
    "shoulder IR moment": Published(67.0, 11.0, "N·m", "Fleisig 1995, just before MER"),
    "elbow varus moment": Published(64.0, 12.0, "N·m", "Fleisig 1995, just before MER"),
    "shoulder compression": Published(1090.0, 110.0, "N", "Fleisig 1995, after release"),
}

# OBP point-of-interest columns that measure the same quantities.
OBP_POI = {
    "pelvis": "max_pelvis_rotational_velo",
    "trunk": "max_torso_rotational_velo",
    "elbow extension": "max_elbow_extension_velo",
    "internal rotation": "max_shoulder_internal_rotational_velo",
    "mer": "max_shoulder_external_rotation",
    "trunk forward at release": "torso_anterior_tilt_br",
    "trunk lateral at release": "torso_lateral_tilt_br",
    "shoulder IR moment": "shoulder_internal_rotation_moment",
    "elbow varus moment": "elbow_varus_moment",
}
