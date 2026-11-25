export type FingerSegments = {
  proximal: number;
  middle: number;
  distal: number;
};

/** Lengths in centimeters. Adult male defaults, used as the torque reference. */
export type HandProfile = {
  /** Wrist to middle fingertip, the scale the photo measurement uses. */
  handLengthCm: number;
  index: FingerSegments;
  middle: FingerSegments;
  ring: FingerSegments;
  pinky: FingerSegments;
  thumb: FingerSegments;
};

export const DEFAULT_HAND: HandProfile = {
  handLengthCm: 19,
  index: { proximal: 4.0, middle: 2.4, distal: 2.0 },
  middle: { proximal: 4.4, middle: 2.6, distal: 2.2 },
  ring: { proximal: 4.1, middle: 2.5, distal: 2.0 },
  pinky: { proximal: 3.2, middle: 2.0, distal: 1.6 },
  thumb: { proximal: 3.4, middle: 3.2, distal: 2.6 },
};

export type FingerId = "index" | "middle" | "ring" | "pinky" | "thumb";

export const FINGER_IDS: FingerId[] = ["index", "middle", "ring", "pinky", "thumb"];

/** Lever arm used for torque. Thumb uses the two distal segments; others use all three. */
export function leverLengthCm(hand: HandProfile, finger: FingerId): number {
  const seg = hand[finger];
  if (finger === "thumb") return seg.middle + seg.distal;
  return seg.proximal + seg.middle + seg.distal;
}

export const REFERENCE_LEVER_CM: Record<FingerId, number> = {
  index: 8.4,
  middle: 9.2,
  ring: 8.6,
  pinky: 6.8,
  thumb: 5.8,
};

/** Scale a finger's segments so its lever length becomes `nextCm`. */
export function withLeverLength(hand: HandProfile, finger: FingerId, nextCm: number): HandProfile {
  const current = leverLengthCm(hand, finger);
  if (!(current > 0.1) || !(nextCm > 0)) return hand;
  const scale = Math.min(1.8, Math.max(0.5, nextCm / current));
  const seg = hand[finger];
  if (finger === "thumb") {
    return { ...hand, thumb: { proximal: seg.proximal, middle: seg.middle * scale, distal: seg.distal * scale } };
  }
  return {
    ...hand,
    [finger]: {
      proximal: seg.proximal * scale,
      middle: seg.middle * scale,
      distal: seg.distal * scale,
    },
  };
}

export function leverScale(hand: HandProfile, finger: FingerId): number {
  const ref = REFERENCE_LEVER_CM[finger];
  const scale = leverLengthCm(hand, finger) / ref;
  return Math.min(1.35, Math.max(0.65, scale));
}

export type Point3 = { x: number; y: number; z: number };

function dist(a: Point3, b: Point3): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

/**
 * MediaPipe Hands, 21 landmarks.
 * 0 wrist, 1–4 thumb, 5–8 index, 9–12 middle, 13–16 ring, 17–20 pinky.
 */
export function handFromLandmarks(
  points: Point3[],
  referenceLengthCm: number,
): { hand: HandProfile; warning: string | null } {
  if (points.length < 21) {
    throw new Error("手のランドマークが21点ありません。");
  }
  if (!(referenceLengthCm > 5 && referenceLengthCm < 30)) {
    throw new Error("基準の長さは 5–30 cm で入れてください。");
  }
  const span = dist(points[0], points[12]);
  if (span < 1e-6) {
    throw new Error("手首と中指先端が重なって見えます。正面に近い写真にしてください。");
  }
  const scale = referenceLengthCm / span;
  const seg = (a: number, b: number) => dist(points[a], points[b]) * scale;
  const hand: HandProfile = {
    handLengthCm: referenceLengthCm,
    thumb: { proximal: seg(1, 2), middle: seg(2, 3), distal: seg(3, 4) },
    index: { proximal: seg(5, 6), middle: seg(6, 7), distal: seg(7, 8) },
    middle: { proximal: seg(9, 10), middle: seg(10, 11), distal: seg(11, 12) },
    ring: { proximal: seg(13, 14), middle: seg(14, 15), distal: seg(15, 16) },
    pinky: { proximal: seg(17, 18), middle: seg(18, 19), distal: seg(19, 20) },
  };

  let warning: string | null = null;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const zs = points.map((p) => p.z);
  const xSpan = Math.max(...xs) - Math.min(...xs);
  const ySpan = Math.max(...ys) - Math.min(...ys);
  const zSpan = Math.max(...zs) - Math.min(...zs);
  if (zSpan > xSpan * 1.4 && zSpan > ySpan * 1.4) {
    warning = "横から撮れているようです。手のひらをカメラに向けた写真の方が長さは安定します。";
  }
  const middle = leverLengthCm(hand, "middle");
  if (middle < 5 || middle > 14) {
    warning = "中指の長さが成人の範囲から外れています。基準長さ（手首から中指の先）を確認してください。";
  }
  return { hand, warning };
}
