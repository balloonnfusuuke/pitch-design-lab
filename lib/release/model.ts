import { BALL_INERTIA_KG_M2, BALL_RADIUS_M, FT_TO_M, MPH_TO_MS, MS_TO_MPH, M_TO_FT } from "../physics/constants";
import { formatClock, omegaFromSpin } from "../physics/spin";
import { aimTarget, velocityTowardTarget, type ReleaseState } from "../physics/trajectory";
import { clamp } from "../physics/vec";
import { FINGER_IDS, leverScale, type FingerId } from "../hand/profile";
import { GRIPS } from "./grips";
import type { FingerForces, LastContact, ReleaseBreakdown, ReleaseInput } from "./types";

const F_CAP_N = 42;

const LAST_CONTACT_TILT: Record<LastContact, number> = {
  even: 0,
  index: -10,
  middle: -2,
  ring: 7,
  pinky: 12,
  thumb: 3,
};

function tangentialForce(input: ReleaseInput): number {
  const grip = GRIPS[input.grip];
  let force = 0;
  for (const id of FINGER_IDS) {
    const percent = clamp(input.fingers[id], 0, 100) / 100;
    force += percent * F_CAP_N * grip.fingerWeights[id] * leverScale(input.hand, id);
  }
  const choke = Math.max(0, (input.fingers.thumb - grip.thumbChokeAbove) / 100) * grip.thumbChoke;
  return Math.max(0, force * grip.tangential * (1 - choke));
}

function contactMs(input: ReleaseInput): number {
  const grip = GRIPS[input.grip];
  const ms =
    grip.contactMs +
    clamp(input.wristExtensionDeg, -30, 30) * 0.025 -
    Math.abs(input.releaseTimingMs) * 0.035;
  return clamp(ms, 4.5, 13);
}

function tiltRightHanded(input: ReleaseInput): number {
  const grip = GRIPS[input.grip];
  const lateral =
    (input.fingers.ring / 100) * grip.ringArm +
    (input.fingers.pinky / 100) * grip.pinkyArm -
    (input.fingers.index / 100) * grip.indexGlove;
  return (
    grip.tiltBase +
    (grip.slotNeutral - input.armSlotDeg) * grip.slotTiltPerDeg +
    (input.pronationDeg - grip.pronationNeutral) * grip.pronationTilt +
    (grip.wristNeutral - input.wristExtensionDeg) * grip.wristTilt +
    LAST_CONTACT_TILT[input.lastContact] * grip.lastContactScale +
    lateral * grip.lateralGainDeg +
    input.releaseTimingMs * grip.timingTiltPerMs
  );
}

function efficiency(input: ReleaseInput): number {
  const grip = GRIPS[input.grip];
  let eff = grip.baseEfficiency;
  eff -= Math.abs(input.pronationDeg - grip.pronationNeutral) * grip.pronationGyro;
  eff -= Math.abs(input.releaseTimingMs) * 0.0032;
  const dominant = dominantFingers(grip.fingerWeights);
  if (dominant.length === 2) {
    const imbalance = Math.abs(input.fingers[dominant[0]] - input.fingers[dominant[1]]) / 100;
    eff -= Math.max(0, imbalance - 0.25) * 0.08;
  }
  return clamp(eff, 0.06, 0.985);
}

function dominantFingers(weights: FingerForces): FingerId[] {
  return [...FINGER_IDS].sort((a, b) => weights[b] - weights[a]).slice(0, 2);
}

function releasePoint(input: ReleaseInput): { x: number; y: number; z: number } {
  const slot = (clamp(input.armSlotDeg, 0, 90) * Math.PI) / 180;
  const heightM = input.pitcherHeightCm / 100;
  const z =
    1.02 +
    0.98 * Math.sin(slot) +
    (heightM - 1.85) * 0.55 -
    input.releaseTimingMs * 0.0032;
  const armSide = input.handedness === "R" ? -1 : 1;
  const x = armSide * (0.18 + 0.58 * Math.cos(slot));
  const y = input.extensionFt * FT_TO_M;
  return { x, y, z: Math.max(0.4, z) };
}

function ballSpeedMs(input: ReleaseInput, forceN: number): number {
  const grip = GRIPS[input.grip];
  let speed = input.armSpeedMph * MPH_TO_MS * grip.speedTransfer;
  speed += (input.wristExtensionDeg - grip.wristSpeedNeutral) * 0.042;
  speed += (forceN / 80) * 0.55;
  const timingLoss = Math.pow(input.releaseTimingMs / 36, 2) * 0.07;
  speed *= 1 - Math.min(0.12, timingLoss);
  return Math.max(8, speed);
}

/**
 * Kinematic release: finger impulse → spin, arm slot and forearm rotation → axis,
 * hand speed and wrist → velocity. Coefficients live in `grips.ts`.
 * Left-handers are the mirror of the same arm-side rules.
 */
export function computeRelease(input: ReleaseInput): { state: ReleaseState; breakdown: ReleaseBreakdown } {
  const grip = GRIPS[input.grip];
  const forceN = tangentialForce(input);
  const contact = contactMs(input);
  const tiltR = tiltRightHanded(input);
  const tiltDeg = input.handedness === "R" ? tiltR : -tiltR;
  const eff = efficiency(input);
  const point = releasePoint(input);
  const speed = ballSpeedMs(input, forceN);

  const omegaImpulse = (forceN * BALL_RADIUS_M * (contact / 1000)) / BALL_INERTIA_KG_M2;
  const handScale = Math.pow(
    Math.max(40, input.armSpeedMph) / grip.speedRefMph,
    grip.rpmHandExponent,
  );
  const rpm = ((omegaImpulse * handScale * 60) / (2 * Math.PI));

  const target = aimTarget(input.aimXIn, input.aimZFt);
  const velocity = velocityTowardTarget(point, target, speed);
  const gyroSign = input.pronationDeg - grip.pronationNeutral >= 0 ? 1 : -1;
  const omega = omegaFromSpin(rpm, eff, tiltDeg, gyroSign, velocity);

  const sswCoeff = Math.max(0, grip.sswCoeff * input.sswScale);
  const sswClock = input.handedness === "R" ? grip.sswClock : -grip.sswClock;

  const armLength = (input.pitcherHeightCm / 100) * 0.36;
  const state: ReleaseState = {
    position: point,
    velocity,
    omega,
    ssw: sswCoeff > 0.0005 ? { coeff: sswCoeff, clockDeg: sswClock } : null,
    tempC: input.tempC,
    elevationM: input.elevationM,
  };

  const breakdown: ReleaseBreakdown = {
    tangentialForceN: forceN,
    contactMs: contact,
    handSpeedMph: input.armSpeedMph,
    ballSpeedMph: speed * MS_TO_MPH,
    spinRpm: rpm,
    spinEfficiency: eff,
    tiltDeg,
    tiltClock: formatClock(tiltDeg, 15),
    releaseXFt: point.x * M_TO_FT,
    releaseYFt: point.y * M_TO_FT,
    releaseZFt: point.z * M_TO_FT,
    armAngularVelRadS: (input.armSpeedMph * MPH_TO_MS) / Math.max(0.4, armLength),
    sswCoeff,
    middleLeverScale: leverScale(input.hand, "middle"),
  };

  return { state, breakdown };
}
