"""Release-window parameters. Numbers with a citation are set from that source.

Rates are shared by every grip. Grip files change pose, seam orientation,
which finger leaves last, and the sign of pronation. Nothing here is adjusted
to hit a Statcast spin rate.
"""

from __future__ import annotations

import math

from pitchlab.balls import DEFAULT_BALL_ID, get_ball

# Active ball is the NPB unified ball. Mass and circumference are the midpoint
# of the shared legal range (see balls.py); pass ball="mlb" to pitch the
# Rawlings preset. 0.145 kg was the previous rounding of that same midpoint.
_BALL = get_ball(DEFAULT_BALL_ID)
BALL_MASS_KG = _BALL.mass_kg
BALL_CIRCUMFERENCE_M = _BALL.circumference_m
BALL_RADIUS_M = _BALL.radius_m

# Fleisig, "The biomechanics of baseball pitching" (UAB dissertation), kinematic
# database Table 2, elite pitchers: maximum shoulder internal-rotation velocity
# 7511 ± 1140 deg/s at 103 ± 2% of the foot-contact-to-release interval
# (just after release). The same peak is quoted as ~7000 deg/s in the abstract.
SHOULDER_IR_DEG_S = 7511.0

# Same table: maximum elbow-extension velocity 2438 ± 270 deg/s at 91 ± 2%.
ELBOW_EXTENSION_DEG_S = 2438.0

# Same table: elbow flexion at ball release 22 ± 4 deg.
ELBOW_FLEXION_AT_RELEASE_DEG = 22.0

# Barrentine, Matsuo, Escamilla, Fleisig, Andrews, J Appl Biomech 14:1–23 (1998).
# Fastball peak wrist-flexion velocity 2954 ± 789 deg/s, timed +3 ± 14 ms
# relative to release. Curveball 2593 ± 573 and changeup 2659 ± 526 are not
# used: one arm program is shared.
WRIST_FLEX_DEG_S = 2954.0

# The same paper's fastball peak forearm pronation is 5218 ± 2496 deg/s at
# +17 ± 11 ms after release (curve 4155, changeup 4114; all are pronation).
# The model uses the fastball peak, shared, centered on that time. A pronation
# rate inside the finger-ball window is not reported, so none is applied there.
PRONATION_FOLLOW_DEG_S = 5218.0
PRONATION_FOLLOW_CENTER_MS = 17.0
PRONATION_FOLLOW_WINDOW_MS = 20.0

# No high-speed study in the sources below reports fingertip angular velocity
# in deg/s. 4200 deg/s is an unpublished smoothstep peak, shared by every grip.
FINGER_FLEX_DEG_S = 4200.0

# Matsuo, Jinji, Hirayama, Nasu, Ozaki, Kumagawa, Sports Biomech 17:180–191
# (2018): the ball begins rolling up the middle fingertip 8 ± 1 ms before release.
CONTACT_MS = 8.0

# Not Stevenson's thumb-off time. Stevenson (1985), cited by Matsuo et al. and
# by Inaba, Yoshioka, Fukashiro (JSME SHD 2019, B-21), has the thumb leaving
# about 6 ms before release. This delay shifts only the designated last finger
# and has no published value.
LAST_CONTACT_DELAY_MS = 4.0

# Male segment lengths used as the internal-rotation moment arm L*sin(elbow).
FOREARM_M = 0.27
HUMERUS_M = 0.30

# Fingertip friction of the active ball (NPB, no application). The MLB rosin
# values live on balls.BALLS["mlb"]. Yamaguchi et al., Commun Mater (2022).
# The slide was ~0.1 m/s on a flat sheet, not a pitch. MuJoCo takes the
# pairwise maximum, so a seam geom at SEAM_FRICTION keeps that coefficient
# against a pad at FRICTION. The seam is a continuous ridge: bump spacing is
# about 2 mm so neighboring 1.5 mm spheres still stand proud of the leather.
# The count and the bump radius are assumed; the height is the ball's.
FRICTION = _BALL.mu_leather
SEAM_FRICTION = _BALL.mu_seam
# 216 samples so every 6th point is exactly the old 36-bump set. The backing
# sphere is paired only with those 36, which is what the adopted pitch felt.
# The pulp nodes are paired with all 216, so the ridge is continuous for them.
SEAM_COUNT = 216
SEAM_BACKING_STRIDE = 6

# Seam height of the active ball. NPB's standard is 0.90 mm; the MLB preset
# uses the Smith / Garrett 2019 mean, 0.80 mm. Bump radius is only how the
# sphere is built (assumed for both balls). Seam width is on the Ball and is
# not this radius.
SEAM_PROTRUSION_M = _BALL.seam_height_m
SEAM_BUMP_RADIUS_M = 0.0015

# Serina, Mote, Rempel, J Biomech (1997) and the 1998 structural model: the pulp
# is compliant below ~1 N and stiffens sharply (82% of the displacement at 5.2 N
# occurs before 1 N). The 1998 model was checked against contact area only for
# 0.25–7 N on a flat plate. No Young's modulus, no contact area at a pitching
# load (~90 N), and no MuJoCo solref are given. solref "0.004 1" is unpublished.
# PAD_RADIUS_M is the outer envelope of the old rigid sphere, kept so the grip
# reach does not move. The pulp itself is a bed of small spheres on radial
# slides (see PULP_*). Those lengths and the 4 N bottoming force are assumed;
# they only copy the published shape (soft, then a hard stop) and clear the
# NPB standard seam (0.90 mm) and the MLB mean (0.80 mm).
PAD_RADIUS_M = 0.009
# Nodes stand this far outside the 9 mm backing and retract until they are
# flush with it. 1.0 mm still clears the NPB standard 0.90 mm; both the stand-off and the
# 4 N bottoming force are assumed (Serina's curve, not a measured millimetre).
PULP_NODE_RADIUS_M = 0.00155
PULP_PROTRUSION_M = 0.00100
PULP_TRAVEL_M = 0.00100
PULP_BOTTOM_N = 4.0
PULP_STIFFNESS_N_PER_M = PULP_BOTTOM_N / PULP_TRAVEL_M
PULP_DAMPING = 1.2
CONTACT_SOLREF = "0.004 1"
CONTACT_SOLIMP = "0.92 0.97 0.001 0.5 2"

# Kinoshita, Obata, Nasu, Kadota, Matsuo, Fleisig, Hum Mov Sci 54:172–181 (2017),
# and the ISBS 2014 precursor: index and middle resultant force second peaks are
# about 90–96 N (middle 89.6 ± 18.5 N at 8 ms before release; index second peak
# 95.6 ± 14.5 N). Shear is about half (middle 47.3 ± 16.4 N). Yamaguchi et al.,
# Front Sports Act Living 2:30 (2020) approximate the normal-force second peak
# as 80 N for a ~31 m/s pitch. The 8 cm moment arm is not in those papers.
# 80 N × 0.08 m = 6.4 N·m, applied as the same MCP cap on every finger.
MCP_TORQUE_NM = 6.4
PIP_TORQUE_NM = 3.2
DIP_TORQUE_NM = 1.6

# Fleisig Table 2: the wrist travels 86 ± 4 cm in the 30 ms after release,
# a 28.7 m/s average while the arm is decelerating. Instantaneous wrist speed
# at release is not in that table (radar ball speed in the same cohort is
# 38.3 ± 0.7 m/s). PROXIMAL_SPEED_MS is only the trunk/shoulder slide.
# It is a few meters per second, in the range of the shoulder's linear motion
# in that cohort, and it is not tuned so the wrist equals 28.7 m/s.
PROXIMAL_SPEED_MS = 4.0

# 2024 Statcast RHP pitch-weighted arm angle, degrees above horizontal
# (baseballsavant pitcher-arm-angles, year=2024, ball_angle weighted by
# n_pitches). 0° is sidearm, 90° is over the top. RHP mean is 37.1°.
ARM_SLOT_DEG = 37.1
# Fleisig Table 2: mean shoulder abduction during acceleration 94 ± 9°.
HUMERUS_ABDUCTION_DEG = 94.0
# Barrentine et al. (1998): maximal external rotation is 24 ± 5 ms before
# release. The simulated acceleration is a little longer so the elbow can
# travel from a cocked flexion to 22° at the published peak rate.
ACCEL_MS = 36.0
# Pronation follow-through. Barrentine's peak pronation is +17 ± 11 ms
# after release, so the clip includes that deceleration.
FOLLOW_MS = 42.0
# Stevenson (1985): the thumb leaves about 6 ms before the ball. Ring and
# pinky produce less force than index and middle (Kinoshita et al. 2017)
# and open on the same clock as the thumb.
THUMB_OFF_MS = 6.0
# Unpublished width of the internal-rotation spike. Fleisig's 7511 deg/s peak
# is at release; holding that rate for the whole acceleration spins the hand
# more than 100° and turns the palm off the ball. A 3.5 ms gaussian keeps the
# peak and limits the arc to about 65°.
IR_SIGMA_S = 0.0035

# Anatomical stops. The volar plate blocks PIP and DIP hyperextension.
# MCP is allowed only a few degrees, so the ball cannot fold a finger
# backward the way an unlimited hinge did.
MCP_EXT_RAD = 0.14
PIP_EXT_RAD = 0.035
DIP_EXT_RAD = 0.035
MCP_FLEX_RAD = 1.57
PIP_FLEX_RAD = 1.75
DIP_FLEX_RAD = 1.40

SETTLE_MS = 2.0
COAST_MS = 12.0

TIMESTEP_S = 5e-5

# Target used only to check the proximal closure. See the comment above.
WRIST_SPEED_TARGET_MS = 0.86 / 0.030


def smoothstep_delta(peak_deg_s: float, duration_s: float) -> float:
    """Angle change (radians) whose smoothstep peaks at `peak_deg_s`."""
    peak = math.radians(peak_deg_s)
    return peak * duration_s / 1.5
