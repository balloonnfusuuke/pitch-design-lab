import { MPH_TO_MS, RUBBER_TO_PLATE_M } from "./constants";
import { omegaFromSpin } from "./spin";
import { simulate, velocityTowardTarget, type ReleaseState, type SimulationResult } from "./trajectory";

/** A release point used for the textbook pitches: RHP, 6.5 ft of extension, ~6.1 ft high. */
export const TEXTBOOK_RELEASE = {
  x: -0.5,
  y: 6.5 * 0.3048,
  z: 1.85,
};

export type ReferenceId = "four-seam" | "sinker" | "slider" | "sweeper" | "curve" | "changeup";

export type ReferencePitch = {
  id: ReferenceId;
  name: string;
  note: string;
  state: ReleaseState;
};

const ZONE_CENTER = { x: 0, y: RUBBER_TO_PLATE_M, z: 2.5 * 0.3048 };

function stateOf(spec: {
  mph: number;
  rpm: number;
  efficiency: number;
  tiltDeg: number;
  ssw?: { coeff: number; clockDeg: number } | null;
}): ReleaseState {
  const position = { ...TEXTBOOK_RELEASE };
  const velocity = velocityTowardTarget(position, ZONE_CENTER, spec.mph * MPH_TO_MS);
  const omega = omegaFromSpin(spec.rpm, spec.efficiency, spec.tiltDeg, 1, velocity);
  return {
    position,
    velocity,
    omega,
    ssw: spec.ssw ?? null,
    tempC: 22,
    elevationM: 0,
  };
}

/**
 * Direct aerodynamic inputs for typical MLB shapes.
 * These do not pass through the finger/arm model — they are the flight-model
 * references the kinematic designer is checked against.
 */
export function referencePitch(id: ReferenceId): ReferencePitch {
  switch (id) {
    case "four-seam":
      return {
        id,
        name: "典型フォーシーム",
        note: "95 mph、2300 rpm、効率 98%、12:30。純粋な空力入力。",
        state: stateOf({ mph: 95, rpm: 2300, efficiency: 0.98, tiltDeg: 15 }),
      };
    case "sinker":
      return {
        id,
        name: "典型シンカー",
        note: "93 mph、2100 rpm、1:30 付近に、小さめの縫い目ウェイク。",
        state: stateOf({
          mph: 93,
          rpm: 2100,
          efficiency: 0.9,
          tiltDeg: 48,
          ssw: { coeff: 0.028, clockDeg: 70 },
        }),
      };
    case "slider":
      return {
        id,
        name: "典型スライダー",
        note: "86 mph、2450 rpm、効率 32% のジャイロスライダー。軸は 8:30。",
        state: stateOf({ mph: 86, rpm: 2450, efficiency: 0.32, tiltDeg: 255 }),
      };
    case "sweeper":
      return {
        id,
        name: "典型スイーパー",
        note: "81 mph、2700 rpm、効率 72%、ほぼ 9:00 の横回転。",
        state: stateOf({ mph: 81, rpm: 2700, efficiency: 0.72, tiltDeg: 272 }),
      };
    case "curve":
      return {
        id,
        name: "典型カーブ",
        note: "79 mph、2650 rpm、効率 84%、7:00 付近のトップスピン。",
        state: stateOf({ mph: 79, rpm: 2650, efficiency: 0.84, tiltDeg: 214 }),
      };
    case "changeup":
      return {
        id,
        name: "典型チェンジアップ",
        note: "84 mph、1700 rpm、1:40。フォーシームより遅く、シュートする。",
        state: stateOf({
          mph: 84,
          rpm: 1700,
          efficiency: 0.88,
          tiltDeg: 50,
          ssw: { coeff: 0.012, clockDeg: 55 },
        }),
      };
  }
}

export function simulateReference(id: ReferenceId): SimulationResult {
  return simulate(referencePitch(id).state);
}

export const REFERENCE_IDS: ReferenceId[] = ["four-seam", "sinker", "slider", "sweeper", "curve", "changeup"];
