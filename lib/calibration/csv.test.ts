import { describe, expect, it } from "vitest";
import { designPitch } from "../release/design";
import { presetInput } from "../release/grips";
import { calibrateMeasured, parsePitchCsv, releaseFromMeasured } from "./csv";

const SAMPLE = `name,velo_mph,spin_rpm,tilt,ivb_in,hb_in,spin_efficiency
四シーム実測,94.5,2320,12:45,17.2,-7.1,0.95
スライダー,85.0,2480,8:30,2.0,6.5,0.34
`;

describe("measured pitch CSV", () => {
  it("parses clocks, inches, and efficiency", () => {
    const rows = parsePitchCsv(SAMPLE);
    expect(rows).toHaveLength(2);
    expect(rows[0].name).toBe("四シーム実測");
    expect(rows[0].tiltDeg).toBeCloseTo(22.5, 1);
    expect(rows[0].hbIn).toBeCloseTo(-7.1, 2);
    expect(rows[1].spinEfficiency).toBeCloseTo(0.34, 2);
  });

  it("converts Statcast pfx feet into inches", () => {
    const rows = parsePitchCsv("pitch,release_speed,spin_rate,pfx_x,pfx_z\nFF,95,2300,-0.6,1.4\n");
    expect(rows[0].hbIn).toBeCloseTo(-7.2, 2);
    expect(rows[0].ivbIn).toBeCloseTo(16.8, 2);
  });

  it("flies a measured pitch and compares it with the current design", () => {
    const measured = parsePitchCsv(SAMPLE)[0];
    const designed = designPitch(presetInput("four-seam"));
    const row = calibrateMeasured(
      measured,
      designed.simulation.metrics,
      designed.simulation.samples[0]?.p ?? { x: -0.4, y: 1.9, z: 1.8 },
      { x: 0, z: 0.76 },
      { tempC: 22, elevationM: 0 },
    );
    expect(row.simulation).not.toBeNull();
    expect(row.simulation!.metrics.releaseSpeedMph).toBeGreaterThan(90);
    expect(row.suggestions.length).toBeGreaterThan(0);
    const state = releaseFromMeasured(
      measured,
      { x: -0.4, y: 1.9, z: 1.8 },
      { x: 0, z: 0.76 },
      { tempC: 22, elevationM: 0 },
    );
    expect(state?.omega.x).toBeTypeOf("number");
  });
});
