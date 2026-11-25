import { designPitch, INTERACTIVE_INTEGRATION, SEARCH_INTEGRATION } from "./design";
import { GRIPS, GRIP_LIST, presetInput } from "./grips";
import type { FingerForces, GripId, ReleaseInput } from "./types";
import type { PitchMetrics } from "../physics/trajectory";
import { clamp } from "../physics/vec";

export type InverseTarget = {
  ivbIn?: number;
  hbIn?: number;
  veloMph?: number;
  plateXIn?: number;
  plateZFt?: number;
};

export type Bottleneck = {
  parameter: keyof Tunable;
  label: string;
  atBound: "low" | "high";
  text: string;
};

export type InverseResult = {
  input: ReleaseInput;
  metrics: PitchMetrics;
  loss: number;
  evals: number;
  summary: string;
  bottlenecks: Bottleneck[];
};

type Tunable = {
  armSlotDeg: number;
  armSpeedMph: number;
  pronationDeg: number;
  wristExtensionDeg: number;
  releaseTimingMs: number;
  index: number;
  middle: number;
  ring: number;
  pinky: number;
  thumb: number;
};

const KEYS = [
  "armSlotDeg",
  "armSpeedMph",
  "pronationDeg",
  "wristExtensionDeg",
  "releaseTimingMs",
  "index",
  "middle",
  "ring",
  "pinky",
  "thumb",
] as const;

const LO: Tunable = {
  armSlotDeg: 8,
  armSpeedMph: 52,
  pronationDeg: -55,
  wristExtensionDeg: -40,
  releaseTimingMs: -28,
  index: 0,
  middle: 0,
  ring: 0,
  pinky: 0,
  thumb: 0,
};

const HI: Tunable = {
  armSlotDeg: 88,
  armSpeedMph: 86,
  pronationDeg: 55,
  wristExtensionDeg: 35,
  releaseTimingMs: 28,
  index: 100,
  middle: 100,
  ring: 100,
  pinky: 100,
  thumb: 100,
};

const LABELS: Record<keyof Tunable, string> = {
  armSlotDeg: "アームスロット",
  armSpeedMph: "腕の速度",
  pronationDeg: "回内・回外",
  wristExtensionDeg: "手首の角度",
  releaseTimingMs: "リリースのタイミング",
  index: "人差し指の力",
  middle: "中指の力",
  ring: "薬指の力",
  pinky: "小指の力",
  thumb: "親指の力",
};

function readTunable(input: ReleaseInput): Tunable {
  return {
    armSlotDeg: input.armSlotDeg,
    armSpeedMph: input.armSpeedMph,
    pronationDeg: input.pronationDeg,
    wristExtensionDeg: input.wristExtensionDeg,
    releaseTimingMs: input.releaseTimingMs,
    index: input.fingers.index,
    middle: input.fingers.middle,
    ring: input.fingers.ring,
    pinky: input.fingers.pinky,
    thumb: input.fingers.thumb,
  };
}

function applyTunable(base: ReleaseInput, tunable: Tunable): ReleaseInput {
  const fingers: FingerForces = {
    index: tunable.index,
    middle: tunable.middle,
    ring: tunable.ring,
    pinky: tunable.pinky,
    thumb: tunable.thumb,
  };
  return { ...base, ...tunable, fingers };
}

function vec(tunable: Tunable): number[] {
  return KEYS.map((key) => tunable[key]);
}

function fromVec(values: number[]): Tunable {
  const tunable = {} as Tunable;
  KEYS.forEach((key, i) => {
    tunable[key] = values[i];
  });
  return tunable;
}

function lossOf(metrics: PitchMetrics, target: InverseTarget): number {
  let loss = 0;
  if (target.ivbIn != null) loss += (metrics.ivbIn - target.ivbIn) ** 2;
  if (target.hbIn != null) loss += (metrics.hbIn - target.hbIn) ** 2;
  if (target.veloMph != null) loss += ((metrics.releaseSpeedMph - target.veloMph) * 0.9) ** 2;
  if (target.plateXIn != null) loss += ((metrics.plateXIn - target.plateXIn) * 0.55) ** 2;
  if (target.plateZFt != null) loss += ((metrics.plateZFt - target.plateZFt) * 8) ** 2;
  return loss;
}

function coordinateSearch(
  loss: (values: number[]) => number,
  start: number[],
  rounds: number,
): { values: number[]; loss: number; evals: number } {
  const values = start.slice();
  const lo = KEYS.map((key) => LO[key]);
  const hi = KEYS.map((key) => HI[key]);
  let current = loss(values);
  let evals = 1;
  for (let round = 0; round < rounds; round++) {
    const frac = 0.5 / 2 ** round;
    for (let i = 0; i < values.length; i++) {
      const span = (hi[i] - lo[i]) * frac;
      let best = values[i];
      let bestLoss = current;
      for (const dir of [-1, -0.4, 0.4, 1]) {
        const trial = clamp(values[i] + dir * span, lo[i], hi[i]);
        if (Math.abs(trial - values[i]) < 1e-6) continue;
        const previous = values[i];
        values[i] = trial;
        const value = loss(values);
        evals += 1;
        if (value + 1e-9 < bestLoss) {
          bestLoss = value;
          best = trial;
        }
        values[i] = previous;
      }
      values[i] = best;
      current = bestLoss;
    }
  }
  return { values, loss: current, evals };
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function jitter(start: number[], rand: () => number, amount: number): number[] {
  return start.map((value, i) => {
    const key = KEYS[i];
    const span = (HI[key] - LO[key]) * amount;
    return clamp(value + (rand() * 2 - 1) * span, LO[key], HI[key]);
  });
}

function evaluateInput(input: ReleaseInput, target: InverseTarget): { loss: number; metrics: PitchMetrics } {
  const designed = designPitch(input, SEARCH_INTEGRATION);
  const metrics = designed.simulation.metrics;
  return { loss: lossOf(metrics, target), metrics };
}

export function inverseDesign(
  target: InverseTarget,
  base: ReleaseInput,
  options: { lockGrip?: boolean; seed?: number } = {},
): InverseResult {
  const grips: GripId[] = options.lockGrip ? [base.grip] : GRIP_LIST.map((grip) => grip.id);
  const rand = mulberry32(options.seed ?? 7);
  let best: { input: ReleaseInput; loss: number; evals: number } | null = null;
  let evals = 0;

  for (const grip of grips) {
    const origin = presetInput(grip, {
      handedness: base.handedness,
      hand: base.hand,
      pitcherHeightCm: base.pitcherHeightCm,
      extensionFt: base.extensionFt,
      aimXIn: target.plateXIn ?? base.aimXIn,
      aimZFt: target.plateZFt ?? base.aimZFt,
      tempC: base.tempC,
      elevationM: base.elevationM,
      sswScale: base.sswScale,
    });
    const starts = [vec(readTunable(origin)), jitter(vec(readTunable(origin)), rand, 0.18)];
    for (const start of starts) {
      const searched = coordinateSearch((values) => {
        return evaluateInput(applyTunable(origin, fromVec(values)), target).loss;
      }, start, 3);
      evals += searched.evals;
      if (!best || searched.loss < best.loss) {
        best = {
          input: applyTunable(origin, fromVec(searched.values)),
          loss: searched.loss,
          evals,
        };
      }
    }
  }

  if (!best) {
    throw new Error("探索を開始できませんでした。");
  }

  const polished = coordinateSearch(
    (values) => evaluateInput(applyTunable(best!.input, fromVec(values)), target).loss,
    vec(readTunable(best.input)),
    2,
  );
  evals += polished.evals;
  const input = applyTunable(best.input, fromVec(polished.values));
  const finalDesign = designPitch(input, INTERACTIVE_INTEGRATION);
  const metrics = finalDesign.simulation.metrics;
  const loss = lossOf(metrics, target);
  const bottlenecks = findBottlenecks(input, target, loss);
  return {
    input,
    metrics,
    loss,
    evals,
    bottlenecks,
    summary: summarize(target, metrics, bottlenecks, input.grip),
  };
}

function findBottlenecks(input: ReleaseInput, target: InverseTarget, loss: number): Bottleneck[] {
  if (loss < 2.2) return [];
  const tunable = readTunable(input);
  const baseLoss = evaluateInput(input, target).loss;
  const found: Bottleneck[] = [];

  for (const key of KEYS) {
    const span = HI[key] - LO[key];
    const eps = span * 0.05;
    const atHigh = tunable[key] >= HI[key] - span * 0.04;
    const atLow = tunable[key] <= LO[key] + span * 0.04;
    if (!atHigh && !atLow) continue;

    const nudged = { ...tunable };
    if (atHigh) nudged[key] = tunable[key] - eps;
    else nudged[key] = tunable[key] + eps;
    const nudgedLoss = evaluateInput(applyTunable(input, nudged), target).loss;
    const wantsOutside = nudgedLoss > baseLoss + 0.15;
    if (!wantsOutside) continue;

    const atBound = atHigh ? "high" : "low";
    found.push({
      parameter: key,
      label: LABELS[key],
      atBound,
      text: bottleneckText(key, atBound, target, input),
    });
  }

  return found.slice(0, 3);
}

function bottleneckText(
  key: keyof Tunable,
  bound: "low" | "high",
  target: InverseTarget,
  input: ReleaseInput,
): string {
  const edge = bound === "high" ? "上限" : "下限";
  if (key === "armSpeedMph" && bound === "high") {
    return "腕の速度が探索範囲の上限です。球速か回転がまだ足りないなら、手先のスピードを上げる振りが先です。";
  }
  if ((key === "index" || key === "middle") && bound === "high") {
    const lever = input.hand.middle.proximal + input.hand.middle.middle + input.hand.middle.distal;
    return `人差し指・中指の力は上限です。中指のレバーは ${lever.toFixed(1)} cm で、同じ力でもトルクはここで頭打ちになります。ボールに指を最後まで残す感覚が課題です。`;
  }
  if (key === "pronationDeg") {
    const want = target.hbIn != null && Math.abs(target.hbIn) > 8 ? "狙った横変化" : "このスピン軸";
    return `回内・回外が可動域の${edge}です。${want}には前腕の回旋がボトルネックになっています。`;
  }
  if (key === "armSlotDeg") {
    return `アームスロットが${edge}です。スピン軸の傾きは腕の角度に強く縛られています。`;
  }
  if (key === "wristExtensionDeg") {
    return `手首の角度が${edge}です。速球なら伸展、カーブなら屈曲をこれ以上はモデルの可動域で出せません。`;
  }
  if (key === "ring" || key === "pinky") {
    return `${LABELS[key]}が${edge}です。チェンジアップやシンカーの抜きは、この指の圧力が限界に当たっています。`;
  }
  return `${LABELS[key]}が探索範囲の${edge}にあり、目標との差をこれ以上は縮められません。`;
}

function summarize(target: InverseTarget, metrics: PitchMetrics, bottlenecks: Bottleneck[], grip: GripId): string {
  const name = GRIPS[grip].name;
  const bits: string[] = [`${name}で探索しました。`];
  if (target.veloMph != null) {
    bits.push(`球速 ${metrics.releaseSpeedMph.toFixed(1)} mph（目標 ${target.veloMph.toFixed(1)}）。`);
  }
  if (target.ivbIn != null) {
    bits.push(`縦変化 ${signed(metrics.ivbIn)} in（目標 ${signed(target.ivbIn)}）。`);
  }
  if (target.hbIn != null) {
    bits.push(`横変化 ${signed(metrics.hbIn)} in（目標 ${signed(target.hbIn)}）。`);
  }
  if (bottlenecks.length === 0) {
    bits.push("目標の近くまで届いています。");
  } else {
    bits.push(`ボトルネック: ${bottlenecks.map((item) => item.label).join("、")}。`);
  }
  return bits.join("");
}

function signed(n: number): string {
  return `${n >= 0 ? "+" : ""}${n.toFixed(1)}`;
}

export const SEARCH_LIMITS = { lo: LO, hi: HI, labels: LABELS };
