import type { FingerForces, GripId, LastContact, ReleaseInput } from "./types";
import { DEFAULT_HAND } from "../hand/profile";

export type GripMark = {
  finger: keyof FingerForces;
  /** Degrees clockwise from the top of the ball, looking along the throw. */
  angleDeg: number;
  radius: number;
};

export type GripSpec = {
  id: GripId;
  name: string;
  short: string;
  blurb: string;
  color: string;
  /** RHP pitcher-view clock of the Magnus force before arm-slot and pronation nudges. */
  tiltBase: number;
  slotNeutral: number;
  /** Degrees of tilt toward arm side per degree the slot drops below neutral. */
  slotTiltPerDeg: number;
  baseEfficiency: number;
  pronationNeutral: number;
  /** Efficiency lost per degree away from the grip's neutral pronation. */
  pronationGyro: number;
  /** Tilt degrees (arm-side positive in the RHP frame) per degree of pronation. */
  pronationTilt: number;
  wristNeutral: number;
  /** Tilt added when the wrist is more flexed than neutral. */
  wristTilt: number;
  wristSpeedNeutral: number;
  /** Share of the normal finger force that is tangential to the ball. */
  tangential: number;
  fingerWeights: FingerForces;
  thumbChokeAbove: number;
  thumbChoke: number;
  contactMs: number;
  /** Ball speed / hand speed, before the wrist term. */
  speedTransfer: number;
  speedRefMph: number;
  rpmHandExponent: number;
  /** Seam-shifted wake, in the RHP clock. Phenomenological. */
  sswClock: number;
  sswCoeff: number;
  lateralGainDeg: number;
  indexGlove: number;
  ringArm: number;
  pinkyArm: number;
  timingTiltPerMs: number;
  lastContactScale: number;
  defaults: {
    armSlotDeg: number;
    armSpeedMph: number;
    pronationDeg: number;
    wristExtensionDeg: number;
    releaseTimingMs: number;
    fingers: FingerForces;
    lastContact: LastContact;
  };
  marks: GripMark[];
};

const fingers = (index: number, middle: number, ring: number, pinky: number, thumb: number): FingerForces => ({
  index,
  middle,
  ring,
  pinky,
  thumb,
});

export const GRIPS: Record<GripId, GripSpec> = {
  "four-seam": {
    id: "four-seam",
    name: "フォーシーム",
    short: "4S",
    blurb: "人差し指と中指を縫い目にかけ、真上に近いバックスピンで伸ばす。",
    color: "#f4f0e4",
    tiltBase: 24,
    slotNeutral: 68,
    slotTiltPerDeg: 0.72,
    baseEfficiency: 0.96,
    pronationNeutral: 6,
    pronationGyro: 0.0065,
    pronationTilt: 0.5,
    wristNeutral: 12,
    wristTilt: 0.22,
    wristSpeedNeutral: 8,
    tangential: 0.84,
    fingerWeights: fingers(1, 1, 0.16, 0.05, 0.05),
    thumbChokeAbove: 78,
    thumbChoke: 0.4,
    contactMs: 7.45,
    speedTransfer: 1.25,
    speedRefMph: 74,
    rpmHandExponent: 0.42,
    sswClock: 25,
    sswCoeff: 0.004,
    lateralGainDeg: 12,
    indexGlove: 0.3,
    ringArm: 0.35,
    pinkyArm: 0.55,
    timingTiltPerMs: 0.32,
    lastContactScale: 0.55,
    defaults: {
      armSlotDeg: 68,
      armSpeedMph: 74.5,
      pronationDeg: 8,
      wristExtensionDeg: 16,
      releaseTimingMs: 0,
      fingers: fingers(84, 86, 20, 10, 36),
      lastContact: "middle",
    },
    marks: [
      { finger: "index", angleDeg: -32, radius: 0.58 },
      { finger: "middle", angleDeg: 16, radius: 0.62 },
      { finger: "ring", angleDeg: 58, radius: 0.78 },
      { finger: "thumb", angleDeg: 168, radius: 0.62 },
    ],
  },
  "two-seam": {
    id: "two-seam",
    name: "ツーシーム",
    short: "2S",
    blurb: "指を揃えて縫い目に沿わせ、アームサイドと沈みを足す。シンカーの入り。",
    color: "#e0b15a",
    tiltBase: 36,
    slotNeutral: 54,
    slotTiltPerDeg: 0.85,
    baseEfficiency: 0.92,
    pronationNeutral: 16,
    pronationGyro: 0.0045,
    pronationTilt: 0.62,
    wristNeutral: 8,
    wristTilt: 0.18,
    wristSpeedNeutral: 6,
    tangential: 0.8,
    fingerWeights: fingers(0.85, 1, 0.55, 0.15, 0.06),
    thumbChokeAbove: 80,
    thumbChoke: 0.3,
    contactMs: 8.3,
    speedTransfer: 1.23,
    speedRefMph: 73,
    rpmHandExponent: 0.4,
    sswClock: 68,
    sswCoeff: 0.02,
    lateralGainDeg: 14,
    indexGlove: 0.2,
    ringArm: 0.7,
    pinkyArm: 0.85,
    timingTiltPerMs: 0.35,
    lastContactScale: 0.7,
    defaults: {
      armSlotDeg: 52,
      armSpeedMph: 73,
      pronationDeg: 20,
      wristExtensionDeg: 8,
      releaseTimingMs: 2,
      fingers: fingers(70, 82, 58, 24, 40),
      lastContact: "ring",
    },
    marks: [
      { finger: "index", angleDeg: -8, radius: 0.5 },
      { finger: "middle", angleDeg: 18, radius: 0.52 },
      { finger: "ring", angleDeg: 42, radius: 0.64 },
      { finger: "thumb", angleDeg: 175, radius: 0.6 },
    ],
  },
  cutter: {
    id: "cutter",
    name: "カッター",
    short: "CT",
    blurb: "フォーシームに近く、人差し指を少し外してグラブサイドへ小さく曲げる。",
    color: "#7eb8c9",
    tiltBase: -8,
    slotNeutral: 66,
    slotTiltPerDeg: 0.28,
    baseEfficiency: 0.84,
    pronationNeutral: -4,
    pronationGyro: 0.007,
    pronationTilt: 0.45,
    wristNeutral: 10,
    wristTilt: 0.2,
    wristSpeedNeutral: 8,
    tangential: 0.78,
    fingerWeights: fingers(1.05, 0.7, 0.12, 0.04, 0.08),
    thumbChokeAbove: 80,
    thumbChoke: 0.35,
    contactMs: 9.5,
    speedTransfer: 1.22,
    speedRefMph: 72,
    rpmHandExponent: 0.38,
    sswClock: -15,
    sswCoeff: 0.008,
    lateralGainDeg: 8,
    indexGlove: 0.45,
    ringArm: 0.2,
    pinkyArm: 0.2,
    timingTiltPerMs: 0.2,
    lastContactScale: 0.8,
    defaults: {
      armSlotDeg: 66,
      armSpeedMph: 72,
      pronationDeg: -2,
      wristExtensionDeg: 12,
      releaseTimingMs: -2,
      fingers: fingers(78, 70, 16, 8, 42),
      lastContact: "index",
    },
    marks: [
      { finger: "index", angleDeg: -48, radius: 0.55 },
      { finger: "middle", angleDeg: 8, radius: 0.58 },
      { finger: "thumb", angleDeg: 150, radius: 0.66 },
    ],
  },
  slider: {
    id: "slider",
    name: "スライダー",
    short: "SL",
    blurb: "軸を打者側に倒し、回転の多くをジャイロに残して小さく鋭く曲げる。",
    color: "#d94a32",
    tiltBase: 250,
    slotNeutral: 58,
    slotTiltPerDeg: 0.35,
    baseEfficiency: 0.36,
    pronationNeutral: -12,
    pronationGyro: 0.004,
    pronationTilt: 0.35,
    wristNeutral: 0,
    wristTilt: 0.45,
    wristSpeedNeutral: 2,
    tangential: 0.92,
    fingerWeights: fingers(1.05, 0.95, 0.2, 0.05, 0.1),
    thumbChokeAbove: 85,
    thumbChoke: 0.25,
    contactMs: 8.4,
    speedTransfer: 1.2,
    speedRefMph: 72,
    rpmHandExponent: 0.3,
    sswClock: 250,
    sswCoeff: 0.008,
    lateralGainDeg: 10,
    indexGlove: 0.4,
    ringArm: 0.15,
    pinkyArm: 0.15,
    timingTiltPerMs: 0.15,
    lastContactScale: 0.45,
    defaults: {
      armSlotDeg: 58,
      armSpeedMph: 73,
      pronationDeg: -14,
      wristExtensionDeg: -4,
      releaseTimingMs: 0,
      fingers: fingers(76, 70, 18, 8, 48),
      lastContact: "index",
    },
    marks: [
      { finger: "index", angleDeg: -70, radius: 0.48 },
      { finger: "middle", angleDeg: -28, radius: 0.5 },
      { finger: "thumb", angleDeg: 140, radius: 0.58 },
    ],
  },
  sweeper: {
    id: "sweeper",
    name: "スイーパー",
    short: "SW",
    blurb: "スロットを少し下げ、横回転を残して大きくグラブサイドへ掃く。",
    color: "#e07a3d",
    tiltBase: 268,
    slotNeutral: 48,
    slotTiltPerDeg: 0.4,
    baseEfficiency: 0.7,
    pronationNeutral: -24,
    pronationGyro: 0.0035,
    pronationTilt: 0.28,
    wristNeutral: -6,
    wristTilt: 0.3,
    wristSpeedNeutral: 0,
    tangential: 0.9,
    fingerWeights: fingers(0.5, 1.1, 0.35, 0.08, 0.08),
    thumbChokeAbove: 82,
    thumbChoke: 0.3,
    contactMs: 11.2,
    speedTransfer: 1.17,
    speedRefMph: 70,
    rpmHandExponent: 0.32,
    sswClock: 275,
    sswCoeff: 0.01,
    lateralGainDeg: 12,
    indexGlove: 0.25,
    ringArm: 0.2,
    pinkyArm: 0.15,
    timingTiltPerMs: 0.12,
    lastContactScale: 0.4,
    defaults: {
      armSlotDeg: 46,
      armSpeedMph: 70,
      pronationDeg: -26,
      wristExtensionDeg: -8,
      releaseTimingMs: 1,
      fingers: fingers(42, 88, 30, 10, 40),
      lastContact: "middle",
    },
    marks: [
      { finger: "middle", angleDeg: -78, radius: 0.46 },
      { finger: "index", angleDeg: -120, radius: 0.62 },
      { finger: "ring", angleDeg: -30, radius: 0.7 },
      { finger: "thumb", angleDeg: 120, radius: 0.6 },
    ],
  },
  curve: {
    id: "curve",
    name: "カーブ",
    short: "CB",
    blurb: "手首を屈曲させ、中指で上から抜いてトップスピンにする。",
    color: "#b089e0",
    tiltBase: 212,
    slotNeutral: 72,
    slotTiltPerDeg: 0.25,
    baseEfficiency: 0.84,
    pronationNeutral: -32,
    pronationGyro: 0.003,
    pronationTilt: 0.22,
    wristNeutral: -22,
    wristTilt: 0.7,
    wristSpeedNeutral: -8,
    tangential: 0.92,
    fingerWeights: fingers(0.6, 1.15, 0.2, 0.05, 0.06),
    thumbChokeAbove: 80,
    thumbChoke: 0.3,
    contactMs: 9.8,
    speedTransfer: 1.17,
    speedRefMph: 68,
    rpmHandExponent: 0.28,
    sswClock: 200,
    sswCoeff: 0.006,
    lateralGainDeg: 8,
    indexGlove: 0.35,
    ringArm: 0.1,
    pinkyArm: 0.1,
    timingTiltPerMs: 0.1,
    lastContactScale: 0.35,
    defaults: {
      armSlotDeg: 72,
      armSpeedMph: 68,
      pronationDeg: -34,
      wristExtensionDeg: -24,
      releaseTimingMs: -4,
      fingers: fingers(48, 90, 16, 6, 44),
      lastContact: "middle",
    },
    marks: [
      { finger: "middle", angleDeg: -20, radius: 0.42 },
      { finger: "index", angleDeg: -55, radius: 0.55 },
      { finger: "thumb", angleDeg: 110, radius: 0.7 },
    ],
  },
  changeup: {
    id: "changeup",
    name: "チェンジアップ",
    short: "CH",
    blurb: "腕は速球のまま、薬指と小指を深く入れて球速と回転を殺す。",
    color: "#8fbf78",
    tiltBase: 46,
    slotNeutral: 62,
    slotTiltPerDeg: 0.7,
    baseEfficiency: 0.86,
    pronationNeutral: 18,
    pronationGyro: 0.004,
    pronationTilt: 0.55,
    wristNeutral: 4,
    wristTilt: 0.16,
    wristSpeedNeutral: 2,
    tangential: 0.58,
    fingerWeights: fingers(0.35, 0.7, 1.05, 0.9, 0.12),
    thumbChokeAbove: 70,
    thumbChoke: 0.2,
    contactMs: 7.2,
    speedTransfer: 1.145,
    speedRefMph: 71,
    rpmHandExponent: 0.35,
    sswClock: 58,
    sswCoeff: 0.02,
    lateralGainDeg: 14,
    indexGlove: 0.15,
    ringArm: 0.75,
    pinkyArm: 0.9,
    timingTiltPerMs: 0.3,
    lastContactScale: 0.65,
    defaults: {
      armSlotDeg: 60,
      armSpeedMph: 71,
      pronationDeg: 22,
      wristExtensionDeg: 4,
      releaseTimingMs: 4,
      fingers: fingers(28, 46, 74, 68, 36),
      lastContact: "ring",
    },
    marks: [
      { finger: "middle", angleDeg: 4, radius: 0.55 },
      { finger: "ring", angleDeg: 36, radius: 0.48 },
      { finger: "pinky", angleDeg: 68, radius: 0.5 },
      { finger: "index", angleDeg: -70, radius: 0.78 },
      { finger: "thumb", angleDeg: 190, radius: 0.55 },
    ],
  },
  splitter: {
    id: "splitter",
    name: "スプリット",
    short: "SP",
    blurb: "人差し指と中指を広げてボールを深く挟み、回転を落として沈ませる。",
    color: "#e58ab0",
    tiltBase: 168,
    slotNeutral: 66,
    slotTiltPerDeg: 0.15,
    baseEfficiency: 0.58,
    pronationNeutral: 4,
    pronationGyro: 0.003,
    pronationTilt: 0.25,
    wristNeutral: 2,
    wristTilt: 0.2,
    wristSpeedNeutral: 4,
    tangential: 0.7,
    fingerWeights: fingers(0.9, 0.9, 0.12, 0.05, 0.1),
    thumbChokeAbove: 88,
    thumbChoke: 0.25,
    contactMs: 9.6,
    speedTransfer: 1.2,
    speedRefMph: 72,
    rpmHandExponent: 0.25,
    sswClock: 175,
    sswCoeff: 0.03,
    lateralGainDeg: 8,
    indexGlove: 0.2,
    ringArm: 0.15,
    pinkyArm: 0.1,
    timingTiltPerMs: 0.12,
    lastContactScale: 0.3,
    defaults: {
      armSlotDeg: 66,
      armSpeedMph: 73,
      pronationDeg: 4,
      wristExtensionDeg: 2,
      releaseTimingMs: 0,
      fingers: fingers(36, 36, 12, 6, 55),
      lastContact: "even",
    },
    marks: [
      { finger: "index", angleDeg: -58, radius: 0.62 },
      { finger: "middle", angleDeg: 42, radius: 0.62 },
      { finger: "thumb", angleDeg: 180, radius: 0.48 },
    ],
  },
};

export const GRIP_LIST: GripSpec[] = [
  GRIPS["four-seam"],
  GRIPS["two-seam"],
  GRIPS.cutter,
  GRIPS.slider,
  GRIPS.sweeper,
  GRIPS.curve,
  GRIPS.changeup,
  GRIPS.splitter,
];

export function presetInput(grip: GripId, keep?: Partial<ReleaseInput>): ReleaseInput {
  const spec = GRIPS[grip];
  const base: ReleaseInput = {
    handedness: "R",
    grip,
    armSlotDeg: spec.defaults.armSlotDeg,
    armSpeedMph: spec.defaults.armSpeedMph,
    pronationDeg: spec.defaults.pronationDeg,
    wristExtensionDeg: spec.defaults.wristExtensionDeg,
    releaseTimingMs: spec.defaults.releaseTimingMs,
    fingers: { ...spec.defaults.fingers },
    lastContact: spec.defaults.lastContact,
    hand: DEFAULT_HAND,
    pitcherHeightCm: 185,
    extensionFt: 6.3,
    aimXIn: 0,
    aimZFt: 2.5,
    tempC: 22,
    elevationM: 0,
    sswScale: 1,
  };
  return {
    ...base,
    ...keep,
    grip,
    fingers: keep?.fingers ? { ...keep.fingers } : base.fingers,
    hand: keep?.hand ?? base.hand,
  };
}
