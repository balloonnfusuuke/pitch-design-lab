import { BALL_AREA_M2, BALL_MASS_KG, BALL_RADIUS_M, GRAVITY_MS2, NATHAN_CL_A, NATHAN_CL_B } from "./constants";
import { cross, dot, mag, normalize, scale, sub, type Vec3 } from "./vec";

/** ISA-like density: 1.225 kg/m³ at 15°C, sea level, exponential scale height 8500 m. */
export function airDensityKgM3(tempC: number, elevationM: number): number {
  const tempK = tempC + 273.15;
  const seaLevel = 1.225 * (288.15 / tempK);
  return seaLevel * Math.exp(-elevationM / 8500);
}

/** Nathan 2020 lift coefficient. S is the transverse spin parameter r ω_T / v. */
export function liftCoefficient(spinParameter: number): number {
  const s = Math.max(0, spinParameter);
  return NATHAN_CL_A * (1 - Math.exp(-NATHAN_CL_B * s));
}

/**
 * Drag coefficient through the band reported for pitched baseballs
 * (about 0.30–0.40 from 70–100 mph in Nathan / Alaways trajectory fits).
 * Unlike CL, there is no single settled closed form, so this is a smooth
 * curve in that band: ~0.33 near 95 mph, a little higher at changeup speed.
 */
export function dragCoefficient(speedMs: number): number {
  const v = Math.max(speedMs, 0.5);
  return 0.3 + 0.1 / (1 + Math.exp((v - 36) / 5));
}

export type AeroFlags = {
  drag: boolean;
  magnus: boolean;
  ssw: boolean;
};

export type SswTerm = {
  /** Added to the Magnus direction. Phenomenological, not a seam CFD model. */
  coeff: number;
  /** Pitcher-view clock, degrees clockwise from 12:00. */
  clockDeg: number;
};

/**
 * Unit vector of a break in the pitcher's clock.
 * 12:00 is +Z, 3:00 is the pitcher's right (−X when +X is the catcher's right).
 */
export function breakDirection(clockDeg: number, velocityHat: Vec3): Vec3 {
  const theta = (clockDeg * Math.PI) / 180;
  let dir: Vec3 = {
    x: -Math.sin(theta),
    y: 0,
    z: Math.cos(theta),
  };
  dir = sub(dir, scale(velocityHat, dot(dir, velocityHat)));
  const m = mag(dir);
  if (m < 1e-8) return { x: 0, y: 0, z: 1 };
  return scale(dir, 1 / m);
}

/** Acceleration of the baseball, including gravity. */
export function acceleration(
  velocity: Vec3,
  omega: Vec3,
  ssw: SswTerm | null,
  rho: number,
  flags: AeroFlags,
): Vec3 {
  const speed = mag(velocity);
  const g: Vec3 = { x: 0, y: 0, z: -GRAVITY_MS2 };
  if (speed < 1e-6) return g;

  const k = (rho * BALL_AREA_M2) / (2 * BALL_MASS_KG);
  let aero: Vec3 = { x: 0, y: 0, z: 0 };

  if (flags.drag) {
    const cd = dragCoefficient(speed);
    aero = scale(velocity, -k * cd * speed);
  }

  if (flags.magnus) {
    const vhat = scale(velocity, 1 / speed);
    const parallel = dot(omega, vhat);
    const trans = sub(omega, scale(vhat, parallel));
    const transMag = mag(trans);
    if (transMag > 1e-8) {
      const spinParameter = (BALL_RADIUS_M * transMag) / speed;
      const cl = liftCoefficient(spinParameter);
      const liftHat = normalize(cross(trans, vhat));
      const magnus = scale(liftHat, k * cl * speed * speed);
      aero = { x: aero.x + magnus.x, y: aero.y + magnus.y, z: aero.z + magnus.z };
    }
  }

  if (flags.ssw && ssw && ssw.coeff > 0) {
    const vhat = scale(velocity, 1 / speed);
    const dir = breakDirection(ssw.clockDeg, vhat);
    const extra = scale(dir, k * ssw.coeff * speed * speed);
    aero = { x: aero.x + extra.x, y: aero.y + extra.y, z: aero.z + extra.z };
  }

  return { x: aero.x + g.x, y: aero.y + g.y, z: aero.z + g.z };
}
