"""Four-seam ball orientation from the published grip, not from a spin search.

The rotation is the one that puts the model's fingertip contacts on the Bourke
seam the way the grip descriptions do. It is recomputed from the finger
directions, so a change in finger spacing moves the seams with the fingers.

Research, used as constraints:

- MLB.com glossary, four-seam fastball: the horseshoe lies across the fingers.
  Both fingers cross the open leather and touch the seams (four contact
  points, each finger on both legs). The edges of the fingers sit slightly
  over the seam.
- efastball.com and Steven Ellis, The Complete Pitcher: the closed end of the
  U points at the thumb and the open end at the ring finger. The thumb pad is
  on smooth leather.
- Wikipedia, four-seam fastball: index and middle lie on or across the
  horseshoe that faces outward; the thumb pad is on leather.
- RPP Baseball: fingertips sit perpendicular on the seams, ½–1 inch apart.
- ACTIVE.com: pads on the stitches, about half an inch apart.

The model's index–middle contact chord is 19.4 mm. Jung, Kim, Shin, Kwon,
Hwang, Lee, Arch Plast Surg 42:753–761 (2015) measured healthy men's DIP
width at 18.8 ± 2.1 mm (index) and 18.6 ± 2.0 mm (middle), so adjacent
fingertips are about 18.7 mm center to center. 19.4 mm is that, plus a
fraction of a millimetre, and it is inside RPP's ½–1 inch. The knuckle
spacing in ``simulate.ROOTS`` was not moved.

Putting both fingertips on one leg leaves the other leg about 12 mm to the
thumb side of the middle-finger axis, outside an 8 mm half-width and also
outside the measured 9.4 mm half-width. Centering the pad on the open
leather brings both legs inside that width (index about 7.8 mm, middle
about 9.5 mm from the contact). That pitch was thrown and not adopted:
spin fell to about 1530 rpm, lateral launch sat on −5°, and finger force
rose past 350 N. The delivery keeps the fingertips on one leg. The closed
end toward the thumb is efastball's primary description.

Assumed, where the texts disagree:

- efastball's alternate (closed end toward the pinky) is not used. RPP treats
  horseshoe-in and horseshoe-out as feel preferences. Driveline (2026) has no
  measured difference yet.
- The Bourke curve is a smooth stand-in for the seam. The rotation that falls
  out is a consequence of this fit, not a measured photograph.
"""

from __future__ import annotations

import math

import numpy as np

from pitchlab.seams import seam_samples

_SEAM_N = 720


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """R such that R @ src ≈ dst. `src` and `dst` are (2, 3)."""
    h = src.T @ dst
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[-1] *= -1
        r = vt.T @ u.T
    return r


def _seam_table(n: int = _SEAM_N):
    pts = seam_samples(n)
    tang = np.array([_unit(pts[(i + 1) % n] - pts[(i - 1) % n]) for i in range(n)])
    partner = np.empty(n, dtype=int)
    for i in range(n):
        d = pts @ pts[i]
        for k in range(-25, 26):
            d[(i + k) % n] = -2.0
        partner[i] = int(np.argmax(d))
    gap = np.linalg.norm(pts - pts[partner], axis=1) * 0.03689
    return pts, tang, partner, gap


def _deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(float(np.asarray(a) @ np.asarray(b)), -1.0, 1.0))))


def _parallel_pairs():
    """Seam samples a finger-pair can stand on: tangents run between the fingers."""
    pts, tang, partner, gap = _seam_table()
    n = len(pts)
    pairs = []
    for i in range(0, n, 3):
        for k in range(12, 96, 2):
            j = (i + k) % n
            if abs(float(tang[i] @ tang[int(partner[i])])) < 0.75:
                continue
            if abs(float(tang[j] @ tang[int(partner[j])])) < 0.75:
                continue
            chord = _unit(pts[j] - pts[i])
            if abs(float(tang[i] @ chord)) < 0.85 or abs(float(tang[j] @ chord)) < 0.85:
                continue
            pairs.append((i, j, _deg(pts[i], pts[j])))
    return pairs


_PAIRS = None


def _pairs():
    global _PAIRS
    if _PAIRS is None:
        _PAIRS = _parallel_pairs()
    return _PAIRS


def four_seam_rotation(
    index_dir: np.ndarray,
    middle_dir: np.ndarray,
    index_axis: np.ndarray,
    middle_axis: np.ndarray,
    thumb_dir: np.ndarray,
    index_prox: np.ndarray | None = None,
    middle_prox: np.ndarray | None = None,
) -> np.ndarray:
    """Ball-local to hand-frame rotation for the four-seam grip.

    Directions are unit vectors in the hand frame (fingers +X, palm normal +Z,
    thumb −Y): from the ball center to each pad, and each finger's distal +X.
    ``index_prox`` / ``middle_prox`` are the proximal pad points. Each finger
    then lies across both legs of one horseshoe.
    """
    index_dir = _unit(np.asarray(index_dir, float))
    middle_dir = _unit(np.asarray(middle_dir, float))
    thumb_dir = _unit(np.asarray(thumb_dir, float))
    if index_prox is None or middle_prox is None:
        index_prox = index_dir
        middle_prox = middle_dir
    index_prox = _unit(np.asarray(index_prox, float))
    middle_prox = _unit(np.asarray(middle_prox, float))

    def distal(axis: np.ndarray, radial: np.ndarray) -> np.ndarray:
        bone = _unit(np.asarray(axis, float))
        return _unit(bone - radial * float(bone @ radial))

    distal_i = distal(index_axis, index_dir)
    distal_m = distal(middle_axis, middle_dir)
    pts, tang, partner, gap = _seam_table()
    ang_f = _deg(index_dir, middle_dir)
    use_prox = float(np.linalg.norm(index_prox - index_dir)) > 1e-6
    # Centering each pad on the open leather does put both legs inside a
    # measured fingertip (index chords about 7.8 mm, middle about 9.5 mm;
    # Jung et al. 2015 DIP half-width 9.4 mm). That pitch was not adopted:
    # spin fell to about 1530 rpm, lateral launch sat on −5°, and the finger
    # force rose. The delivery keeps both fingertips on one leg.
    best = None
    for i, j, ang_s in _pairs():
        if abs(ang_s - ang_f) > 2.5:
            continue
        for a, b in ((i, j), (j, i)):
            r = _kabsch(np.stack([pts[a], pts[b]]), np.stack([index_dir, middle_dir]))
            err = _deg(r @ pts[a], index_dir) + _deg(r @ pts[b], middle_dir)
            if use_prox:
                err += 2.0 * _deg(r @ pts[int(partner[a])], index_prox)
                err += 2.0 * _deg(r @ pts[int(partner[b])], middle_prox)
            cross = abs(float((r @ tang[a]) @ distal_i)) + abs(float((r @ tang[b]) @ distal_m))
            g_back = float(gap[(a - 4) % len(pts)])
            g_fwd = float(gap[(a + 4) % len(pts)])
            close_dir = (r @ tang[a]) * (-1.0 if g_back < g_fwd else 1.0)
            # Hand −Y is the thumb. The closed end of the U points that way.
            closed_thumb = -float(close_dir[1])
            thumb_deg = math.degrees(math.acos(float(np.clip(np.max(pts @ (r.T @ thumb_dir)), -1.0, 1.0))))
            score = err + 20.0 * cross + (0.0 if closed_thumb > 0.2 else 15.0) + max(0.0, 18.0 - thumb_deg) * 0.4
            key = (score, a, b)
            if best is None or key < best[0]:
                best = (key, r)
    if best is None:
        raise RuntimeError("no four-seam orientation put both fingers across the horseshoe")
    return best[1]


def grip_report(
    rotation: np.ndarray,
    index_dir,
    middle_dir,
    index_axis,
    middle_axis,
    thumb_dir,
    index_prox=None,
    middle_prox=None,
) -> dict[str, float]:
    """Angles that say whether the published constraints held. For tests and the log."""
    pts, tang, partner, gap = _seam_table()
    r = rotation

    def distal(axis, radial):
        bone = _unit(np.asarray(axis, float))
        radial = _unit(np.asarray(radial, float))
        return _unit(bone - radial * float(bone @ radial)), radial

    out = {}
    for name, direction, axis in (
        ("index", index_dir, index_axis),
        ("middle", middle_dir, middle_axis),
        ("thumb", thumb_dir, thumb_dir),
    ):
        along, radial = distal(axis, direction)
        local = r.T @ radial
        i = int(np.argmax(pts @ local))
        out[f"{name}_seam_deg"] = math.degrees(math.acos(float(np.clip((pts @ local)[i], -1.0, 1.0))))
        out[f"{name}_cross"] = abs(float((r @ tang[i]) @ along))
        # The U this pad actually spans: both legs as close as possible.
        # The single nearest seam can be a different part of the curve.
        span = None
        for k in range(0, len(pts), 2):
            p = int(partner[k])
            d1 = float(np.linalg.norm(pts[k] - local)) * 36.89
            d2 = float(np.linalg.norm(pts[p] - local)) * 36.89
            worst = max(d1, d2)
            if span is None or worst < span[0]:
                span = (worst, min(d1, d2), max(d1, d2), float(gap[k]) * 1000.0)
        out[f"{name}_near_mm"] = span[1]
        out[f"{name}_other_mm"] = span[2]
        out[f"{name}_span_gap_mm"] = span[3]
        if name == "index":
            other = r @ pts[int(partner[i])]
            out["other_leg_proximal"] = float(_unit(other - radial) @ (-along))
            out["horseshoe_gap_mm"] = float(gap[i]) * 1000.0
    for name, prox in (("index", index_prox), ("middle", middle_prox)):
        if prox is None:
            continue
        local = r.T @ _unit(np.asarray(prox, float))
        i = int(np.argmax(pts @ local))
        out[f"{name}_prox_seam_deg"] = math.degrees(math.acos(float(np.clip((pts @ local)[i], -1.0, 1.0))))
    return out
