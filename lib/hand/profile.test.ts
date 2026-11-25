import { describe, expect, it } from "vitest";
import { handFromLandmarks, leverLengthCm, type Point3 } from "./profile";

function point(x: number, y: number, z = 0): Point3 {
  return { x, y, z };
}

describe("hand measurement", () => {
  it("scales landmarks so the wrist-to-middle length matches the reference", () => {
    const points: Point3[] = Array.from({ length: 21 }, () => point(0, 0, 0));
    points[0] = point(0, 0, 0);
    points[12] = point(0, 0.19, 0);
    points[5] = point(-0.02, 0.08, 0);
    points[6] = point(-0.025, 0.12, 0);
    points[7] = point(-0.028, 0.145, 0);
    points[8] = point(-0.03, 0.165, 0);
    points[9] = point(0, 0.09, 0);
    points[10] = point(0, 0.13, 0);
    points[11] = point(0, 0.16, 0);
    points[1] = point(0.02, 0.04, 0);
    points[2] = point(0.04, 0.06, 0);
    points[3] = point(0.055, 0.08, 0);
    points[4] = point(0.07, 0.09, 0);
    points[13] = point(0.02, 0.085, 0);
    points[14] = point(0.025, 0.12, 0);
    points[15] = point(0.028, 0.145, 0);
    points[16] = point(0.03, 0.16, 0);
    points[17] = point(0.04, 0.08, 0);
    points[18] = point(0.05, 0.11, 0);
    points[19] = point(0.055, 0.13, 0);
    points[20] = point(0.06, 0.145, 0);

    const { hand } = handFromLandmarks(points, 19);
    expect(hand.handLengthCm).toBe(19);
    expect(leverLengthCm(hand, "middle")).toBeGreaterThan(6);
    expect(leverLengthCm(hand, "middle")).toBeLessThan(12);
  });

  it("rejects a missing reference length", () => {
    const points = Array.from({ length: 21 }, () => point(0, 0, 0));
    points[12] = point(0, 1, 0);
    expect(() => handFromLandmarks(points, 0)).toThrow();
  });
});
