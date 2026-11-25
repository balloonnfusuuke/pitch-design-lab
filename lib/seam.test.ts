import { describe, expect, it } from "vitest";
import { baseballSeamPoint, baseballSeamSamples } from "./seam";

describe("baseball seam", () => {
  it("stays on the sphere and closes", () => {
    const points = baseballSeamSamples(240);
    for (const [x, y, z] of points) {
      expect(Math.abs(Math.hypot(x, y, z) - 1)).toBeLessThan(1e-6);
    }
    const a = baseballSeamPoint(0);
    const b = baseballSeamPoint(Math.PI * 4);
    expect(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])).toBeLessThan(1e-6);
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const zs = points.map((p) => p[2]);
    expect(Math.max(...xs) - Math.min(...xs)).toBeGreaterThan(1.2);
    expect(Math.max(...ys) - Math.min(...ys)).toBeGreaterThan(1.2);
    expect(Math.max(...zs) - Math.min(...zs)).toBeGreaterThan(0.6);
  });
});
