import { breakDirection } from "./aero";
import { add, cross, dot, mag, normalize, scale, sub, type Vec3 } from "./vec";

export function wrap360(deg: number): number {
  return ((deg % 360) + 360) % 360;
}

/** Clock face, pitcher looking toward home. 12:00 is pure backspin. */
export function formatClock(degClockwiseFrom12: number, stepMinutes = 1): string {
  const d = wrap360(degClockwiseFrom12);
  const rawMinutes = (d / 360) * 12 * 60;
  const snapped = Math.round(rawMinutes / stepMinutes) * stepMinutes;
  const wrapped = ((snapped % (12 * 60)) + 12 * 60) % (12 * 60);
  let hours = Math.floor(wrapped / 60);
  const minutes = wrapped % 60;
  if (hours === 0) hours = 12;
  return `${hours}:${minutes.toString().padStart(2, "0")}`;
}

export function parseClock(text: string): number | null {
  const match = text.trim().match(/^(\d{1,2})\s*:\s*(\d{1,2})$/);
  if (!match) return null;
  let hours = Number(match[1]);
  const minutes = Number(match[2]);
  if (hours === 12) hours = 0;
  if (hours < 0 || hours > 11 || minutes < 0 || minutes >= 60) return null;
  return ((hours * 60 + minutes) / (12 * 60)) * 360;
}

export function spinEfficiency(omega: Vec3, velocity: Vec3): number {
  const om = mag(omega);
  const vm = mag(velocity);
  if (om < 1e-10 || vm < 1e-10) return 0;
  const parallel = Math.min(1, Math.abs(dot(omega, velocity)) / (om * vm));
  return Math.sqrt(1 - parallel * parallel);
}

export function spinRpm(omega: Vec3): number {
  return (mag(omega) * 60) / (2 * Math.PI);
}

/**
 * Pitcher-view tilt of the Magnus force, degrees clockwise from 12:00.
 * Recovered from ω × v, so it matches the force the aerodynamic model applies.
 */
export function tiltDegrees(omega: Vec3, velocity: Vec3): number {
  const vm = mag(velocity);
  if (vm < 1e-8 || mag(omega) < 1e-8) return 0;
  const vhat = scale(velocity, 1 / vm);
  const force = sub(cross(omega, vhat), scale(vhat, dot(cross(omega, vhat), vhat)));
  if (mag(force) < 1e-10) return 0;

  let up = sub({ x: 0, y: 0, z: 1 }, scale(vhat, dot({ x: 0, y: 0, z: 1 }, vhat)));
  if (mag(up) < 1e-8) up = { x: 0, y: 0, z: 1 };
  up = normalize(up);

  let right = sub({ x: -1, y: 0, z: 0 }, scale(vhat, dot({ x: -1, y: 0, z: 0 }, vhat)));
  right = sub(right, scale(up, dot(right, up)));
  if (mag(right) < 1e-8) right = { x: -1, y: 0, z: 0 };
  right = normalize(right);

  const upComp = dot(force, up);
  const rightComp = dot(force, right);
  return wrap360((Math.atan2(rightComp, upComp) * 180) / Math.PI);
}

/**
 * Build a lab-frame spin vector.
 * `tiltDeg` is the pitcher-view clock of the Magnus force (0 = 12:00 = +Z).
 * Efficiency is |ω transverse| / |ω|. Gyro spin lies along the velocity.
 */
export function omegaFromSpin(
  rpm: number,
  efficiency: number,
  tiltDeg: number,
  gyroSign: number,
  velocity: Vec3,
): Vec3 {
  const vhat = normalize(velocity);
  if (mag(vhat) < 1e-8) return { x: 0, y: 0, z: 0 };
  const breakDir = breakDirection(tiltDeg, vhat);
  const transHat = normalize(cross(vhat, breakDir));
  const eff = Math.min(1, Math.max(0, efficiency));
  const omegaMag = (Math.max(0, rpm) * 2 * Math.PI) / 60;
  const gyro = Math.sqrt(Math.max(0, 1 - eff * eff)) * Math.sign(gyroSign || 1);
  return add(scale(transHat, omegaMag * eff), scale(vhat, omegaMag * gyro));
}
