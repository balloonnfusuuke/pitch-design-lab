"""Segment anthropometry for the full-body model.

de Leva, J Biomech 29:1223–1230 (1996), Table 4, males (stature 1741 mm):
segment length, mass fraction, center of mass (% of length from the proximal
end as de Leva defines it), and radii of gyration (% of length) about the
sagittal (anteroposterior), transverse (mediolateral), and longitudinal axes.
The trunk is de Leva's three-part trunk: upper (suprasternale to xiphoid),
middle (xiphoid to omphalion), lower (omphalion to the hip-joint line).

Shoulder girdle: clavicle 0.156 kg and scapula 0.705 kg from Veeger et al.,
J Biomech 24:615–629 (1991). One cadaver, so it is a fraction of an assumed
75 kg body and taken out of the upper trunk.
"""

from __future__ import annotations

from dataclasses import dataclass

DE_LEVA_STATURE_M = 1.741


@dataclass(frozen=True)
class Segment:
    length_m: float  # at 1741 mm stature
    mass_frac: float
    com_frac: float  # from de Leva's proximal end
    r_sag: float  # about the anteroposterior axis
    r_trans: float  # about the mediolateral axis
    r_long: float


DE_LEVA = {
    "head": Segment(0.2033, 0.0694, 0.5976, 0.362, 0.376, 0.312),
    "upper_trunk": Segment(0.1707, 0.1596, 0.2999, 0.716, 0.454, 0.659),
    "middle_trunk": Segment(0.2155, 0.1633, 0.4502, 0.482, 0.383, 0.468),
    "lower_trunk": Segment(0.1457, 0.1117, 0.6115, 0.615, 0.551, 0.587),
    "upper_arm": Segment(0.2817, 0.0271, 0.5772, 0.285, 0.269, 0.158),
    "forearm": Segment(0.2689, 0.0162, 0.4574, 0.276, 0.265, 0.121),
    "hand": Segment(0.0862, 0.0061, 0.7900, 0.628, 0.513, 0.401),
    "thigh": Segment(0.4222, 0.1416, 0.4095, 0.329, 0.329, 0.149),
    "shank": Segment(0.4340, 0.0433, 0.4459, 0.255, 0.249, 0.103),
    "foot": Segment(0.2581, 0.0137, 0.4415, 0.257, 0.245, 0.124),
}
GIRDLE_MASS_FRAC = (0.156 + 0.705) / 75.0
# Ankle joint height 0.039 H is Winter's (Biomechanics and Motor Control of
# Human Movement). Hip-joint-center spacing 0.154 H is the OBP subjects' mean.
ANKLE_HEIGHT_FRAC = 0.039
HIP_WIDTH_FRAC = 0.154
GLOVE_KG = 0.60


@dataclass(frozen=True)
class BodyParams:
    """The pitcher being simulated. Lengths scale with height, masses with mass."""

    height_m: float = 1.85
    mass_kg: float = 88.0

    @property
    def scale(self) -> float:
        return self.height_m / DE_LEVA_STATURE_M

    def length(self, name: str) -> float:
        return DE_LEVA[name].length_m * self.scale

    def mass(self, name: str) -> float:
        return DE_LEVA[name].mass_frac * self.mass_kg

    def inertia(self, name: str, mass: float | None = None) -> tuple[float, float, float]:
        """Diagonal inertia (x mediolateral, y anteroposterior, z longitudinal), kg·m²."""
        s = DE_LEVA[name]
        m = self.mass(name) if mass is None else mass
        L = self.length(name)
        return (m * (s.r_trans * L) ** 2, m * (s.r_sag * L) ** 2, m * (s.r_long * L) ** 2)

    @property
    def girdle_mass(self) -> float:
        return GIRDLE_MASS_FRAC * self.mass_kg

    @property
    def hip_half_m(self) -> float:
        return 0.5 * HIP_WIDTH_FRAC * self.height_m

    @property
    def ankle_height_m(self) -> float:
        return ANKLE_HEIGHT_FRAC * self.height_m
