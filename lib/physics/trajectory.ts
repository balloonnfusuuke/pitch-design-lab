import { acceleration, airDensityKgM3, type AeroFlags, type SswTerm } from "./aero";
import {
  FT_TO_M,
  GRAVITY_MS2,
  M_TO_FT,
  M_TO_IN,
  MS_TO_MPH,
  RUBBER_TO_PLATE_M,
  SPIN_DECAY_TAU_S,
} from "./constants";
import { spinEfficiency, spinRpm, tiltDegrees, formatClock } from "./spin";
import { add, lerp, mag, normalize, scale, type Vec3 } from "./vec";

export type ReleaseState = {
  position: Vec3;
  velocity: Vec3;
  /** Lab-frame spin, rad/s. */
  omega: Vec3;
  ssw: SswTerm | null;
  tempC: number;
  elevationM: number;
};

export type Sample = {
  t: number;
  p: Vec3;
  v: Vec3;
};

export type PlateCrossing = {
  t: number;
  p: Vec3;
  v: Vec3;
};

export type PitchMetrics = {
  releaseSpeedMph: number;
  plateSpeedMph: number;
  spinRpm: number;
  spinEfficiency: number;
  /** Pitcher-view degrees clockwise from 12:00. */
  tiltDeg: number;
  /** Nearest 15 minutes, the way a Rapsodo face usually reads. */
  tiltClock: string;
  tiltClockExact: string;
  /** Induced vertical break, inches. Positive is ride (up vs a spinless pitch). */
  ivbIn: number;
  /** Horizontal break, inches. Positive is the catcher's right (1B side). */
  hbIn: number;
  plateXIn: number;
  plateZFt: number;
  releaseXFt: number;
  releaseZFt: number;
  flightTimeS: number;
  /** Positive means the ball is descending at the plate. */
  approachAngleDeg: number;
  totalDropIn: number;
};

export type SimulationResult = {
  samples: Sample[];
  plate: PlateCrossing;
  spinlessSamples: Sample[];
  spinlessPlate: PlateCrossing;
  metrics: PitchMetrics;
};

const FULL_FLAGS: AeroFlags = { drag: true, magnus: true, ssw: true };
const SPINLESS_FLAGS: AeroFlags = { drag: true, magnus: false, ssw: false };

type Phase = { p: Vec3; v: Vec3 };

function rk4(phase: Phase, dt: number, omega: Vec3, ssw: SswTerm | null, rho: number, flags: AeroFlags): Phase {
  const deriv = (p: Phase) => ({
    p: p.v,
    v: acceleration(p.v, omega, ssw, rho, flags),
  });
  const k1 = deriv(phase);
  const k2 = deriv({ p: add(phase.p, scale(k1.p, dt / 2)), v: add(phase.v, scale(k1.v, dt / 2)) });
  const k3 = deriv({ p: add(phase.p, scale(k2.p, dt / 2)), v: add(phase.v, scale(k2.v, dt / 2)) });
  const k4 = deriv({ p: add(phase.p, scale(k3.p, dt)), v: add(phase.v, scale(k3.v, dt)) });
  const mix = (a: Vec3, b: Vec3, c: Vec3, d: Vec3) => ({
    x: (a.x + 2 * b.x + 2 * c.x + d.x) / 6,
    y: (a.y + 2 * b.y + 2 * c.y + d.y) / 6,
    z: (a.z + 2 * b.z + 2 * c.z + d.z) / 6,
  });
  return {
    p: add(phase.p, scale(mix(k1.p, k2.p, k3.p, k4.p), dt)),
    v: add(phase.v, scale(mix(k1.v, k2.v, k3.v, k4.v), dt)),
  };
}

export type IntegrateOptions = {
  dt?: number;
  sampleDt?: number;
  plateY?: number;
  maxTime?: number;
  storeSamples?: boolean;
  flags?: AeroFlags;
  /** When false, spin magnitude is constant. */
  spinDecay?: boolean;
};

function integrate(release: ReleaseState, options: IntegrateOptions): { samples: Sample[]; plate: PlateCrossing } {
  const dt = options.dt ?? 0.0005;
  const sampleDt = options.sampleDt ?? 0.004;
  const plateY = options.plateY ?? RUBBER_TO_PLATE_M;
  const maxTime = options.maxTime ?? 1.25;
  const store = options.storeSamples !== false;
  const flags = options.flags ?? FULL_FLAGS;
  const rho = airDensityKgM3(release.tempC, release.elevationM);
  const useSsw = flags.ssw ? release.ssw : null;

  let phase: Phase = { p: { ...release.position }, v: { ...release.velocity } };
  let omega = { ...release.omega };
  let t = 0;
  let nextSample = 0;
  const samples: Sample[] = [];
  const push = () => {
    if (store) samples.push({ t, p: { ...phase.p }, v: { ...phase.v } });
  };
  push();

  while (t < maxTime && phase.p.y < plateY) {
    const prev = phase;
    const prevT = t;
    phase = rk4(phase, dt, omega, useSsw, rho, flags);
    t += dt;
    if (options.spinDecay !== false) {
      omega = scale(omega, Math.exp(-dt / SPIN_DECAY_TAU_S));
    }
    if (phase.p.y >= plateY && prev.p.y < plateY) {
      const dy = phase.p.y - prev.p.y;
      const u = dy === 0 ? 1 : (plateY - prev.p.y) / dy;
      const plate: PlateCrossing = {
        t: prevT + dt * u,
        p: lerp(prev.p, phase.p, u),
        v: lerp(prev.v, phase.v, u),
      };
      if (store) samples.push({ t: plate.t, p: { ...plate.p }, v: { ...plate.v } });
      return { samples, plate };
    }
    if (store && t + 1e-9 >= nextSample) {
      push();
      nextSample += sampleDt;
    }
  }

  const plate: PlateCrossing = { t, p: { ...phase.p }, v: { ...phase.v } };
  return { samples, plate };
}

export function simulate(release: ReleaseState, options: IntegrateOptions = {}): SimulationResult {
  const full = integrate(release, { ...options, flags: options.flags ?? FULL_FLAGS });
  const spinless = integrate(
    { ...release, omega: { x: 0, y: 0, z: 0 }, ssw: null },
    { ...options, flags: SPINLESS_FLAGS },
  );
  return {
    samples: full.samples,
    plate: full.plate,
    spinlessSamples: spinless.samples,
    spinlessPlate: spinless.plate,
    metrics: metricsOf(release, full.plate, spinless.plate),
  };
}

export function metricsOf(release: ReleaseState, plate: PlateCrossing, spinless: PlateCrossing): PitchMetrics {
  const v0 = mag(release.velocity);
  const vp = mag(plate.v);
  const tilt = tiltDegrees(release.omega, release.velocity);
  const horiz = Math.max(plate.v.y, 1e-6);
  return {
    releaseSpeedMph: v0 * MS_TO_MPH,
    plateSpeedMph: vp * MS_TO_MPH,
    spinRpm: spinRpm(release.omega),
    spinEfficiency: spinEfficiency(release.omega, release.velocity),
    tiltDeg: tilt,
    tiltClock: formatClock(tilt, 15),
    tiltClockExact: formatClock(tilt, 1),
    ivbIn: (plate.p.z - spinless.p.z) * M_TO_IN,
    hbIn: (plate.p.x - spinless.p.x) * M_TO_IN,
    plateXIn: plate.p.x * M_TO_IN,
    plateZFt: plate.p.z * M_TO_FT,
    releaseXFt: release.position.x * M_TO_FT,
    releaseZFt: release.position.z * M_TO_FT,
    flightTimeS: plate.t,
    approachAngleDeg: (Math.atan2(-plate.v.z, horiz) * 180) / Math.PI,
    totalDropIn: (release.position.z - plate.p.z) * M_TO_IN,
  };
}

/**
 * Initial velocity of `speed` m/s whose drag+gravity (spinless) path
 * finishes near `target`. A short fixed-point iteration accounts for drop.
 */
export function velocityTowardTarget(release: Vec3, target: Vec3, speed: number): Vec3 {
  let velocity: Vec3 = { x: 0, y: Math.max(speed, 1), z: 0 };
  for (let i = 0; i < 4; i++) {
    const dx = target.x - release.x;
    const dy = target.y - release.y;
    const horizSpeed = Math.max(Math.hypot(velocity.x, velocity.y), 1);
    const t = Math.hypot(dx, dy) / horizSpeed;
    const drop = 0.5 * GRAVITY_MS2 * t * t * 1.12;
    const dz = target.z + drop - release.z;
    velocity = scale(normalize({ x: dx, y: dy, z: dz }), speed);
  }
  return velocity;
}

export function sampleAtTime(samples: Sample[], time: number): Sample | null {
  if (samples.length === 0) return null;
  if (time <= samples[0].t) return samples[0];
  const last = samples[samples.length - 1];
  if (time >= last.t) return last;
  let lo = 0;
  let hi = samples.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (samples[mid].t < time) lo = mid;
    else hi = mid;
  }
  const a = samples[lo];
  const b = samples[hi];
  const u = b.t === a.t ? 0 : (time - a.t) / (b.t - a.t);
  return { t: time, p: lerp(a.p, b.p, u), v: lerp(a.v, b.v, u) };
}

/** Interpolate the flight to a plane of constant y (toward home). */
export function sampleAtY(samples: Sample[], y: number): Sample | null {
  if (samples.length < 2) return null;
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1];
    const b = samples[i];
    if ((a.p.y <= y && b.p.y >= y) || (a.p.y >= y && b.p.y <= y)) {
      const dy = b.p.y - a.p.y;
      const u = dy === 0 ? 0 : (y - a.p.y) / dy;
      return { t: a.t + (b.t - a.t) * u, p: lerp(a.p, b.p, u), v: lerp(a.v, b.v, u) };
    }
  }
  return null;
}

export function separationInches(a: Sample[], b: Sample[], y: number): number | null {
  const pa = sampleAtY(a, y);
  const pb = sampleAtY(b, y);
  if (!pa || !pb) return null;
  return mag({ x: pa.p.x - pb.p.x, y: 0, z: pa.p.z - pb.p.z }) * M_TO_IN;
}

/** Commit-point distance in front of the plate, where a hitter has to decide. */
export const TUNNEL_BEFORE_PLATE_M = 6.4;

export function tunnelPlaneY(plateY = RUBBER_TO_PLATE_M): number {
  return plateY - TUNNEL_BEFORE_PLATE_M;
}

export function aimTarget(aimXIn: number, aimZFt: number, plateY = RUBBER_TO_PLATE_M): Vec3 {
  return { x: aimXIn * 0.0254, y: plateY, z: aimZFt * FT_TO_M };
}
