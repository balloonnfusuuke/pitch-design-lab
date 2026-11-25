"""Minimum-jerk joint curves through published means, weighted by their SDs.

Each joint angle is sampled on a uniform grid. A target is a linear function
of those samples: an angle at a time, a rate at a time, or a zero
acceleration (the time of a published peak). The fit minimizes

    sum ((target - mean) / sd)^2  +  lam * sum (jerk * dt^3 / scale)^2

which is a weighted linear least-squares problem. The jerk term follows the
minimum-jerk model of Flash & Hogan, J Neurosci 5:1688–1703 (1985). It only
decides the shape between targets; it is not a published pitching curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Target:
    kind: str  # "angle", "rate", or "accel"
    t: float  # seconds
    mean: float  # degrees, deg/s, or deg/s^2
    sd: float
    source: str = ""


@dataclass
class FittedCurve:
    name: str
    t0: float
    dt: float
    angle: np.ndarray  # degrees
    rate: np.ndarray = field(default_factory=lambda: np.zeros(0))  # deg/s
    targets: tuple[Target, ...] = ()
    jerk_scale: float = 1.0e6
    lam: float = 1.0

    def __post_init__(self) -> None:
        self.rate = np.gradient(self.angle, self.dt)

    def at(self, t: float) -> tuple[float, float]:
        """Radians and rad/s, linearly interpolated."""
        x = (t - self.t0) / self.dt
        n = len(self.angle)
        if x <= 0:
            return math.radians(float(self.angle[0])), math.radians(float(self.rate[0]))
        if x >= n - 1:
            return math.radians(float(self.angle[-1])), math.radians(float(self.rate[-1]))
        i = int(x)
        f = x - i
        a = self.angle[i] * (1 - f) + self.angle[i + 1] * f
        r = self.rate[i] * (1 - f) + self.rate[i + 1] * f
        return math.radians(float(a)), math.radians(float(r))

    def residuals(self) -> list[tuple[Target, float]]:
        """Fitted value for each target, in the target's units."""
        out = []
        for target in self.targets:
            q, qd = self.at(target.t)
            if target.kind == "angle":
                out.append((target, math.degrees(q)))
            elif target.kind == "rate":
                out.append((target, math.degrees(qd)))
            else:
                _, r1 = self.at(target.t + self.dt)
                _, r0 = self.at(target.t - self.dt)
                out.append((target, math.degrees((r1 - r0) / (2 * self.dt))))
        return out


def _row(n: int, t0: float, dt: float, kind: str, t: float) -> np.ndarray:
    """Linear functional on the grid samples."""
    row = np.zeros(n)
    x = (t - t0) / dt
    i = int(math.floor(x))
    i = min(max(i, 2), n - 4)
    f = x - i
    if kind == "angle":
        row[i] += 1 - f
        row[i + 1] += f
    elif kind == "rate":
        # Central differences at i and i+1, interpolated.
        for j, w in ((i, 1 - f), (i + 1, f)):
            row[j + 1] += w / (2 * dt)
            row[j - 1] -= w / (2 * dt)
    elif kind == "accel":
        for j, w in ((i, 1 - f), (i + 1, f)):
            row[j + 1] += w / (dt * dt)
            row[j] -= 2 * w / (dt * dt)
            row[j - 1] += w / (dt * dt)
    else:
        raise ValueError(kind)
    return row


def fit(
    name: str,
    targets: list[Target],
    t0: float,
    t1: float,
    dt: float = 0.0005,
    lam: float = 1.0,
    jerk_scale: float = 1.0e6,
) -> FittedCurve:
    """Solve for the grid angles. `jerk_scale` is deg/s^3, the size of a normal jerk."""
    n = int(round((t1 - t0) / dt)) + 1
    rows = []
    rhs = []
    for target in targets:
        rows.append(_row(n, t0, dt, target.kind, target.t) / target.sd)
        rhs.append(target.mean / target.sd)
    a = np.array(rows)
    b = np.array(rhs)
    # Third differences, scaled so the penalty per sample is O(1) at jerk_scale.
    d3 = np.zeros((n - 3, n))
    for i in range(n - 3):
        d3[i, i : i + 4] = (-1.0, 3.0, -3.0, 1.0)
    d3 /= dt ** 3 * jerk_scale
    weight = math.sqrt(lam / (n - 3))
    full = np.vstack([a, d3 * weight])
    rhs_full = np.concatenate([b, np.zeros(n - 3)])
    solution, *_ = np.linalg.lstsq(full, rhs_full, rcond=None)
    return FittedCurve(
        name=name, t0=t0, dt=dt, angle=solution, targets=tuple(targets), jerk_scale=jerk_scale, lam=lam
    )


def target_rows(curve: FittedCurve) -> tuple[np.ndarray, np.ndarray]:
    """SD-scaled target rows and right-hand side, as used by `fit`."""
    n = len(curve.angle)
    rows = [_row(n, curve.t0, curve.dt, t.kind, t.t) / t.sd for t in curve.targets]
    return np.array(rows), np.array([t.mean / t.sd for t in curve.targets])


def jerk_rows(curve: FittedCurve) -> np.ndarray:
    """The weighted third-difference rows of `fit`."""
    n = len(curve.angle)
    d3 = np.zeros((n - 3, n))
    for i in range(n - 3):
        d3[i, i : i + 4] = (-1.0, 3.0, -3.0, 1.0)
    return d3 / (curve.dt ** 3 * curve.jerk_scale) * math.sqrt(curve.lam / (n - 3))
