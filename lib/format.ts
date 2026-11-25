import type { PitchMetrics } from "./physics/trajectory";
import type { Handedness } from "./release/types";

export function signed(value: number, digits = 1): string {
  const text = value.toFixed(digits);
  return value > 0 ? `+${text}` : text;
}

export function mphAndKmh(mph: number): { mph: string; kmh: string } {
  return { mph: mph.toFixed(1), kmh: (mph * 1.609344).toFixed(1) };
}

export function inchesAndCm(inches: number): { inches: string; cm: string } {
  return { inches: signed(inches, 1), cm: signed(inches * 2.54, 1) };
}

/** Positive result is arm-side break, in inches. */
export function armSideInches(metrics: Pick<PitchMetrics, "hbIn">, hand: Handedness): number {
  return hand === "R" ? -metrics.hbIn : metrics.hbIn;
}

export function breakWords(metrics: Pick<PitchMetrics, "hbIn">, hand: Handedness): string {
  const arm = armSideInches(metrics, hand);
  const magnitude = Math.abs(metrics.hbIn).toFixed(1);
  if (Math.abs(arm) < 0.8) return "ほぼ縦だけ";
  return arm > 0 ? `アームサイド ${magnitude} in` : `グラブサイド ${magnitude} in`;
}

export function plateCall(
  metrics: Pick<PitchMetrics, "plateXIn" | "plateZFt">,
  zone: { bottomFt: number; topFt: number },
  ballRadiusIn = 1.45,
): "strike" | "ball" {
  const half = 17 / 2 + ballRadiusIn;
  const zIn = metrics.plateZFt * 12;
  if (Math.abs(metrics.plateXIn) <= half && zIn >= zone.bottomFt * 12 && zIn <= zone.topFt * 12) {
    return "strike";
  }
  return "ball";
}
