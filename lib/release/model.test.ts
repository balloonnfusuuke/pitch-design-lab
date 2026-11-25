import { describe, expect, it } from "vitest";
import { DEFAULT_HAND, leverLengthCm } from "../hand/profile";
import { armSideInches } from "../format";
import { designPitch } from "./design";
import { presetInput } from "./grips";
import type { GripId } from "./types";

function designed(grip: GripId, hand: "R" | "L" = "R") {
  return designPitch(presetInput(grip, { handedness: hand }));
}

describe("preset release model", () => {
  it("puts a four-seamer in a big-league window", () => {
    const { simulation, breakdown } = designed("four-seam");
    const m = simulation.metrics;
    expect(m.releaseSpeedMph).toBeGreaterThan(90);
    expect(m.releaseSpeedMph).toBeLessThan(99);
    expect(m.spinRpm).toBeGreaterThan(2000);
    expect(m.spinRpm).toBeLessThan(2650);
    expect(m.spinEfficiency).toBeGreaterThan(0.85);
    expect(m.ivbIn).toBeGreaterThan(13);
    expect(m.ivbIn).toBeLessThan(23);
    expect(armSideInches(m, "R")).toBeGreaterThan(2);
    expect(armSideInches(m, "R")).toBeLessThan(14);
    expect(breakdown.tangentialForceN).toBeGreaterThan(30);
    expect(breakdown.releaseZFt).toBeGreaterThan(5.4);
    expect(breakdown.releaseZFt).toBeLessThan(6.8);
  });

  it("separates the pitch menu", () => {
    const fb = designed("four-seam").simulation.metrics;
    const si = designed("two-seam").simulation.metrics;
    const ct = designed("cutter").simulation.metrics;
    const sl = designed("slider").simulation.metrics;
    const sw = designed("sweeper").simulation.metrics;
    const cb = designed("curve").simulation.metrics;
    const ch = designed("changeup").simulation.metrics;
    const sp = designed("splitter").simulation.metrics;

    expect(si.ivbIn).toBeLessThan(fb.ivbIn - 2);
    expect(armSideInches(si, "R")).toBeGreaterThan(armSideInches(fb, "R"));
    expect(armSideInches(ct, "R")).toBeLessThan(armSideInches(fb, "R"));
    expect(armSideInches(sw, "R")).toBeLessThan(-8);
    expect(Math.abs(armSideInches(sw, "R"))).toBeGreaterThan(Math.abs(armSideInches(sl, "R")));
    expect(cb.ivbIn).toBeLessThan(-4);
    expect(ch.releaseSpeedMph).toBeLessThan(fb.releaseSpeedMph - 6);
    expect(ch.spinRpm).toBeLessThan(fb.spinRpm - 300);
    expect(sp.spinRpm).toBeLessThan(1600);
    expect(sl.spinEfficiency).toBeLessThan(sw.spinEfficiency);
  });

  it("responds to arm speed, finger force, pronation, and finger length", () => {
    const base = presetInput("four-seam");
    const slow = designPitch({ ...base, armSpeedMph: base.armSpeedMph - 8 }).simulation.metrics;
    const fast = designPitch({ ...base, armSpeedMph: base.armSpeedMph + 6 }).simulation.metrics;
    expect(fast.releaseSpeedMph).toBeGreaterThan(slow.releaseSpeedMph + 8);

    const soft = designPitch({
      ...base,
      fingers: { ...base.fingers, index: 30, middle: 30 },
    }).simulation.metrics;
    const hard = designPitch(base).simulation.metrics;
    expect(hard.spinRpm).toBeGreaterThan(soft.spinRpm + 400);
    expect(hard.ivbIn).toBeGreaterThan(soft.ivbIn + 2);

    const pronated = designPitch({ ...base, pronationDeg: 40 }).simulation.metrics;
    expect(pronated.spinEfficiency).toBeLessThan(hard.spinEfficiency - 0.08);
    expect(armSideInches(pronated, "R")).toBeGreaterThan(armSideInches(hard, "R"));

    const shortHand = {
      ...DEFAULT_HAND,
      middle: { proximal: 3.2, middle: 1.8, distal: 1.5 },
      index: { proximal: 3.0, middle: 1.7, distal: 1.4 },
    };
    expect(leverLengthCm(shortHand, "middle")).toBeLessThan(7);
    const short = designPitch({ ...base, hand: shortHand }).simulation.metrics;
    expect(short.spinRpm).toBeLessThan(hard.spinRpm - 150);
  });

  it("mirrors a left-hander", () => {
    const right = designed("four-seam", "R").simulation.metrics;
    const left = designed("four-seam", "L").simulation.metrics;
    expect(left.releaseXFt).toBeGreaterThan(0);
    expect(right.releaseXFt).toBeLessThan(0);
    expect(left.hbIn).toBeGreaterThan(1);
    expect(right.hbIn).toBeLessThan(-1);
    expect(Math.abs(left.ivbIn - right.ivbIn)).toBeLessThan(1.5);
  });

  it("drops the release when the slot drops", () => {
    const over = designPitch(presetInput("four-seam", { armSlotDeg: 80 })).breakdown;
    const side = designPitch(presetInput("four-seam", { armSlotDeg: 18 })).breakdown;
    expect(over.releaseZFt).toBeGreaterThan(side.releaseZFt + 1.2);
    expect(Math.abs(side.releaseXFt)).toBeGreaterThan(Math.abs(over.releaseXFt));
  });
});
