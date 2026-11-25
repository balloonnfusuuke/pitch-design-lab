import { describe, expect, it } from "vitest";
import { designPitch } from "./design";
import { trainingHints } from "./hints";
import { inverseDesign } from "./inverse";
import { presetInput } from "./grips";

describe("inverse design", () => {
  it("holds a sweeper target when that grip is locked", () => {
    const goal = designPitch(presetInput("sweeper")).simulation.metrics;
    const result = inverseDesign(
      { ivbIn: goal.ivbIn, hbIn: goal.hbIn, veloMph: goal.releaseSpeedMph },
      presetInput("sweeper"),
      { lockGrip: true, seed: 3 },
    );
    expect(Math.abs(result.metrics.hbIn - goal.hbIn)).toBeLessThan(3);
    expect(Math.abs(result.metrics.ivbIn - goal.ivbIn)).toBeLessThan(3);
    expect(Math.abs(result.metrics.releaseSpeedMph - goal.releaseSpeedMph)).toBeLessThan(2.5);
    expect(result.input.grip).toBe("sweeper");
  });

  it("switches grip when a sweeper target is searched freely", () => {
    const goal = designPitch(presetInput("sweeper")).simulation.metrics;
    const result = inverseDesign(
      { ivbIn: goal.ivbIn, hbIn: goal.hbIn, veloMph: goal.releaseSpeedMph },
      presetInput("four-seam"),
      { lockGrip: false, seed: 4 },
    );
    expect(Math.abs(result.metrics.hbIn - goal.hbIn)).toBeLessThan(4);
    expect(["sweeper", "slider", "cutter"]).toContain(result.input.grip);
  });

  it("names a bound when 30 inches of ride is out of reach", () => {
    const result = inverseDesign(
      { ivbIn: 30, hbIn: -3, veloMph: 98 },
      presetInput("four-seam"),
      { lockGrip: true, seed: 1 },
    );
    expect(result.metrics.ivbIn).toBeGreaterThan(14);
    expect(result.metrics.ivbIn).toBeLessThan(28);
    expect(result.bottlenecks.length).toBeGreaterThan(0);
    expect(result.summary.length).toBeGreaterThan(8);
  });
});

describe("training hints", () => {
  it("flags a soft-fingered fastball", () => {
    const input = presetInput("four-seam", {
      fingers: { index: 25, middle: 25, ring: 10, pinky: 5, thumb: 30 },
    });
    const hints = trainingHints(input, designPitch(input).simulation.metrics);
    expect(hints.some((hint) => hint.id === "ff-fingers" || hint.id === "ff-spin")).toBe(true);
  });

  it("flags a slow arm", () => {
    const input = presetInput("four-seam", { armSpeedMph: 58 });
    const hints = trainingHints(input, designPitch(input).simulation.metrics);
    expect(hints.some((hint) => hint.id === "ff-arm")).toBe(true);
  });
});
