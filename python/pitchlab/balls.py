"""MLB and NPB balls. The default pitch uses the NPB unified ball (Mizuno).

Legal size and mass are the same range in both leagues, and no sample mean
for the current NPB ball was found, so both presets use that midpoint.
What differs, and is used in the contact, is the seam height and the
fingertip friction.

Not used in the finger contact: the bat coefficient of restitution, the
seam width, and the stitch count. MuJoCo's solref is not that COR. The seam
curve is still Bourke's formula for both balls; neither league has published
a measured centerline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# 5 oz and 5.25 oz, written 141.7 g and 148.8 g in the Japanese rule.
_MASS_KG = (0.1417 + 0.1488) / 2.0
# 9 in and 9.25 in. The centimetre figures in the rule are that range, rounded.
_CIRCUMFERENCE_M = 9.125 * 0.0254


@dataclass(frozen=True)
class Ball:
    id: str
    name: str
    mass_kg: float
    circumference_m: float
    # Height of the seam above the leather. This is the protrusion of the ridge.
    seam_height_m: float
    # Width across the stitch pair. Stored, not used as the sphere radius.
    seam_width_m: float | None
    # Fingertip μ actually used for this preset, and the condition it came from.
    mu_leather: float
    mu_seam: float
    mu_condition: str
    # No-application μ from the same paper, so the two balls can be compared
    # without the rosin that was measured only on MLB.
    mu_leather_bare: float
    mu_seam_bare: float
    cor_target: float | None
    stitches: int | None
    # Short source notes. `role` is "measured", "standard", or "substitute".
    sources: tuple[tuple[str, str, str], ...]

    @property
    def radius_m(self) -> float:
        return self.circumference_m / (2.0 * math.pi)


# Yamaguchi, Nasu, Masani, Commun Mater 3:92 (2022): on an MLB ball, rosin
# raised μ by 23.9% without a seam and 27.0% with a seam. Those percentages
# were not measured on NPB leather. Multiplying the NPB no-application
# midpoints by them is a substitute (see the "npb-rosin" preset).
ROSIN_LEATHER_GAIN = 1.239
ROSIN_SEAM_GAIN = 1.270


def _size_sources() -> tuple[tuple[str, str, str], ...]:
    return (
        (
            "mass",
            "substitute",
            "公認野球規則 3.01 and NPB 統一試合球規則 (2015-02-03): 141.7–148.8 g. "
            "No sample mean for the current Mizuno ball was found, so both presets use the midpoint.",
        ),
        (
            "circumference",
            "substitute",
            "Same rules: 9–9.25 in (22.9–23.5 cm). Midpoint 9.125 in. Not a measured NPB mean.",
        ),
    )


BALLS: dict[str, Ball] = {
    "npb": Ball(
        id="npb",
        name="NPB 統一球（ミズノ）",
        mass_kg=_MASS_KG,
        circumference_m=_CIRCUMFERENCE_M,
        # NPB 試合使用球規則, 2015-02-03: 縫い目高さの標準は 1/28 in = 0.9 mm.
        # This is the manufacturing standard, not a sample mean.
        seam_height_m=0.00090,
        seam_width_m=0.0080,
        # Yamaguchi, Nasu, Masani, Commun Mater 3:92 (2022), no application.
        # NPB+rosin was not measured. Midpoints of the 95% CIs, matched to the
        # stated +16.6% (with seam) and +23.1% (without) against the MLB
        # no-application CIs in the same paragraph. The extracted sentence
        # labels both CIs "without seam"; the percentages decide which is which.
        mu_leather=0.9585,
        mu_seam=1.0165,
        mu_condition="no application (rosin on NPB leather was not measured)",
        mu_leather_bare=0.9585,
        mu_seam_bare=1.0165,
        cor_target=0.4134,
        stitches=108,
        sources=_size_sources()
        + (
            (
                "seam height",
                "standard",
                "NPB 2015-02-03 第2条: 標準 1/28 inch (0.9 mm). Also the 2011-02 change "
                "from 1.1 mm (NPB 2010-08-23 / Mizuno). Not a measured mean.",
            ),
            (
                "seam width",
                "standard",
                "NPB 2015-02-03: 標準 5/16 inch (0.8 cm). Not used as the contact-sphere radius.",
            ),
            (
                "friction",
                "measured",
                "Yamaguchi et al. 2022, no-application, CI midpoint. "
                "With seam 0.876–1.157; without seam 0.770–1.147.",
            ),
            (
                "COR",
                "standard",
                "NPB 2015-02-03: 反発係数の目標値 0.4134. Not applied to the finger contact; "
                "that test is a bat rebound, and solref is unchanged.",
            ),
            (
                "stitches",
                "standard",
                "NPB 2015-02-03: 108. The model still draws a continuous ridge, not 108 stitches.",
            ),
            (
                "seam curve",
                "substitute",
                "Bourke's curve, shared with the MLB preset. No published NPB centerline.",
            ),
        ),
    ),
    "npb-rosin": Ball(
        id="npb-rosin",
        name="NPB 統一球＋ロジン（代用）",
        mass_kg=_MASS_KG,
        circumference_m=_CIRCUMFERENCE_M,
        seam_height_m=0.00090,
        seam_width_m=0.0080,
        # Substitute: NPB no-application midpoint times the MLB rosin gain.
        # 0.9585 × 1.239 = 1.1876; 1.0165 × 1.270 = 1.2910.
        mu_leather=round(0.9585 * ROSIN_LEATHER_GAIN, 4),
        mu_seam=round(1.0165 * ROSIN_SEAM_GAIN, 4),
        mu_condition="substitute: NPB no-application × MLB rosin gain (23.9% / 27.0%)",
        mu_leather_bare=0.9585,
        mu_seam_bare=1.0165,
        cor_target=0.4134,
        stitches=108,
        sources=_size_sources()
        + (
            (
                "seam height",
                "standard",
                "Same NPB 2015-02-03 standard as the no-application preset, 0.90 mm.",
            ),
            (
                "friction",
                "substitute",
                "Yamaguchi et al. 2022 measured the rosin gain on MLB leather only: "
                "+23.9% without seam, +27.0% with seam. NPB+rosin was not in that "
                "paper. These coefficients are the NPB no-application CI midpoints "
                "multiplied by those gains. Yamaguchi et al., Front Sports Act Living "
                "2:30 (2020) found that dry rosin on NPB leather did not reliably "
                "raise μ. Not a measurement of this ball.",
            ),
            (
                "seam curve",
                "substitute",
                "Bourke's curve, same as the other presets.",
            ),
        ),
    ),
    "mlb": Ball(
        id="mlb",
        name="MLB（ローリングス想定）",
        mass_kg=_MASS_KG,
        circumference_m=_CIRCUMFERENCE_M,
        # Smith / Garrett, baseballaero.com Post 32 (2019-09-15): 72 new 2019
        # MLB baseballs, mean seam height 0.0314 in = 0.80 mm.
        seam_height_m=0.00080,
        seam_width_m=None,
        # Rosin, the condition this model has been pitching with.
        mu_leather=0.965,
        mu_seam=1.107,
        mu_condition="rosin on the index finger",
        mu_leather_bare=(0.647 + 0.910) / 2.0,
        mu_seam_bare=(0.714 + 1.029) / 2.0,
        cor_target=None,
        stitches=108,
        sources=_size_sources()
        + (
            (
                "seam height",
                "measured",
                "Smith / Garrett 2019, 72 new MLB balls, mean 0.80 mm (SD 0.08 mm).",
            ),
            (
                "seam width",
                "missing",
                "Not in the sources used here. The contact sphere radius stays the assumed 1.5 mm.",
            ),
            (
                "friction",
                "measured",
                "Yamaguchi et al. 2022, rosin, CI midpoint: without seam 0.923–1.007, "
                "with seam 1.028–1.186. No-application midpoints are stored beside these "
                "and are not what this preset pitches with.",
            ),
            (
                "COR",
                "missing",
                "The NPB 0.4134 test is not the MLB specification. Left unset rather than copied.",
            ),
            (
                "stitches",
                "standard",
                "108, the same construction the NPB rule names. Not a separate MLB measurement here.",
            ),
            (
                "seam curve",
                "substitute",
                "Bourke's curve. Not a scan of a Rawlings ball.",
            ),
        ),
    ),
}

DEFAULT_BALL_ID = "npb"


def get_ball(ball: str | Ball | None = None) -> Ball:
    if isinstance(ball, Ball):
        return ball
    return BALLS[DEFAULT_BALL_ID if ball is None else ball]


def contact_mu(ball: str | Ball | None, friction: float | None) -> tuple[Ball, float, float]:
    """Leather μ and seam μ. An explicit `friction` scales the seam with it."""
    spec = get_ball(ball)
    mu = spec.mu_leather if friction is None else float(friction)
    seam = spec.mu_seam * (mu / spec.mu_leather)
    return spec, mu, seam
