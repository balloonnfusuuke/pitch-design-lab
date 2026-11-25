"""Zero-lag low-pass used for motion data and for model loads."""

from __future__ import annotations

import math

import numpy as np


def lowpass(x: np.ndarray, fs: float, fc: float = 20.0) -> np.ndarray:
    """Zero-lag Butterworth low-pass: a 2nd-order section run forward and back (4th order).

    OBP filtered its kinematics and loads at 20 Hz; model loads get the same
    filter before their peaks are compared.
    """
    k = math.tan(math.pi * fc / fs)
    norm = 1.0 / (1.0 + math.sqrt(2.0) * k + k * k)
    b0 = k * k * norm
    b = (b0, 2.0 * b0, b0)
    a1 = 2.0 * (k * k - 1.0) * norm
    a2 = (1.0 - math.sqrt(2.0) * k + k * k) * norm

    def run(u: np.ndarray) -> np.ndarray:
        y = np.zeros_like(u)
        x1 = x2 = u[0]
        y1 = y2 = u[0]
        for i, xi in enumerate(u):
            yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a1 * y1 - a2 * y2
            x2, x1 = x1, xi
            y2, y1 = y1, yi
            y[i] = yi
        return y

    pad = min(len(x) - 1, int(fs / fc))
    ext = np.concatenate([2 * x[0] - x[pad:0:-1], x, 2 * x[-1] - x[-2 : -pad - 2 : -1]])
    y = run(run(ext)[::-1])[::-1]
    return y[pad : pad + len(x)]
