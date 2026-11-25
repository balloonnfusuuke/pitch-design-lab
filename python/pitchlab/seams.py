"""Baseball seam curve (Paul Bourke), same construction as lib/seam.ts."""

from __future__ import annotations

import math

import numpy as np


def seam_point(t: float) -> np.ndarray:
    a = 0.4
    lift = math.pi / 2 - (math.pi / 2 - a) * math.cos(t)
    x = math.sin(lift) * math.cos(t / 2 + a * math.sin(2 * t))
    y = math.sin(lift) * math.sin(t / 2 + a * math.sin(2 * t))
    z = math.cos(lift)
    v = np.array([x, y, z], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def seam_samples(count: int = 40) -> np.ndarray:
    ts = np.linspace(0, 4 * math.pi, count, endpoint=False)
    return np.stack([seam_point(float(t)) for t in ts], axis=0)
