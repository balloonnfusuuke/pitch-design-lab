import type { FingerId, HandProfile } from "../hand/profile";

export type { FingerId, HandProfile };

export type GripId =
  | "four-seam"
  | "two-seam"
  | "cutter"
  | "slider"
  | "sweeper"
  | "curve"
  | "changeup"
  | "splitter";

export type Handedness = "R" | "L";

export type LastContact = FingerId | "even";

export type FingerForces = Record<FingerId, number>;

export type ReleaseInput = {
  handedness: Handedness;
  grip: GripId;
  /** 0° sidearm, 90° over the top. */
  armSlotDeg: number;
  /** Speed of the hand at release, not the ball. */
  armSpeedMph: number;
  /**
   * Positive is pronation (spins the axis toward arm side and adds gyro).
   * Negative is supination. The sign is mirrored for a left-hander, so
   * positive always means "toward this pitcher's arm side".
   */
  pronationDeg: number;
  /** Positive extension (fastball whip). Negative flexion (curve hook). */
  wristExtensionDeg: number;
  /** Negative is early, positive is late, milliseconds from the intended release. */
  releaseTimingMs: number;
  fingers: FingerForces;
  lastContact: LastContact;
  hand: HandProfile;
  pitcherHeightCm: number;
  extensionFt: number;
  /** Catcher's right is positive, inches from the middle of the plate. */
  aimXIn: number;
  /** Height of the intended location, feet. */
  aimZFt: number;
  tempC: number;
  elevationM: number;
  /** Multiplier on the grip's seam-shifted-wake coefficient. 0 turns it off. */
  sswScale: number;
};

export type ReleaseBreakdown = {
  tangentialForceN: number;
  contactMs: number;
  handSpeedMph: number;
  ballSpeedMph: number;
  spinRpm: number;
  spinEfficiency: number;
  tiltDeg: number;
  tiltClock: string;
  releaseXFt: number;
  releaseYFt: number;
  releaseZFt: number;
  armAngularVelRadS: number;
  sswCoeff: number;
  middleLeverScale: number;
};
