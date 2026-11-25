"""MediaPipe Pose sequences as joint-center landmarks for `imitate`.

This does not read video. The 33-point BlazePose skeletons have to be in the
model frame already (+Y toward home, +Z up, +X to a right-hander's throwing
side, meters); camera calibration and lifting image points to 3D are a
separate step. Left-handers are mirrored (x → −x) before this call.

MediaPipe's hip, shoulder, and hand points are surface estimates, not the
functional joint centers OBP reports, so the reconstructed segment frames
are rougher than OBP's.
"""

from __future__ import annotations

import numpy as np

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


def landmarks_from_mediapipe(frames: np.ndarray) -> dict[str, np.ndarray]:
    """(n, 33, 3) right-handed sequence → the landmark names of `obp.POINTS`.

    thorax_prox is the mid-shoulder point, thorax_dist the point halfway to
    the mid-hip, thorax_ap 10 cm toward the throwing shoulder (OBP's
    thorax_ap is a thorax-fixed point on that side).
    """
    p = np.asarray(frames, dtype=float)
    if p.ndim != 3 or p.shape[1] < 29 or p.shape[2] != 3:
        raise ValueError("expected (frames, 33, 3)")
    mid_sh = 0.5 * (p[:, LEFT_SHOULDER] + p[:, RIGHT_SHOULDER])
    mid_hip = 0.5 * (p[:, LEFT_HIP] + p[:, RIGHT_HIP])
    lateral = p[:, RIGHT_SHOULDER] - p[:, LEFT_SHOULDER]
    lateral /= np.maximum(np.linalg.norm(lateral, axis=1, keepdims=True), 1e-9)
    return {
        "thorax_prox": mid_sh,
        "thorax_dist": 0.5 * (mid_sh + mid_hip),
        "thorax_ap": mid_sh + 0.10 * lateral,
        "shoulder_jc": p[:, RIGHT_SHOULDER],
        "elbow_jc": p[:, RIGHT_ELBOW],
        "wrist_jc": p[:, RIGHT_WRIST],
        "hand_jc": 0.5 * (p[:, RIGHT_INDEX] + p[:, RIGHT_PINKY]),
        "glove_shoulder_jc": p[:, LEFT_SHOULDER],
        "glove_elbow_jc": p[:, LEFT_ELBOW],
        "glove_wrist_jc": p[:, LEFT_WRIST],
        "glove_hand_jc": 0.5 * (p[:, LEFT_INDEX] + p[:, LEFT_PINKY]),
        "lead_hip": p[:, LEFT_HIP],
        "rear_hip": p[:, RIGHT_HIP],
        "lead_knee_jc": p[:, LEFT_KNEE],
        "rear_knee_jc": p[:, RIGHT_KNEE],
        "lead_ankle_jc": p[:, LEFT_ANKLE],
        "rear_ankle_jc": p[:, RIGHT_ANKLE],
    }
