import { describe, expect, it } from "vitest";
import { airDensityKgM3, dragCoefficient, liftCoefficient } from "./aero";
import { BALL_RADIUS_M, NATHAN_CL_A, NATHAN_CL_B } from "./constants";
import { referencePitch, simulateReference, type ReferenceId } from "./reference";
import { formatClock, omegaFromSpin, parseClock, spinEfficiency, tiltDegrees } from "./spin";
import { simulate, type ReleaseState } from "./trajectory";
import { mag } from "./vec";

function range(name: string, value: number, lo: number, hi: number) {
  expect(value, `${name}=${value.toFixed(2)} expected ${lo}..${hi}`).toBeGreaterThanOrEqual(lo);
  expect(value, `${name}=${value.toFixed(2)} expected ${lo}..${hi}`).toBeLessThanOrEqual(hi);
}

describe("Nathan coefficients", () => {
  it("matches the 2020 lift fit at a four-seam spin parameter", () => {
    const rpm = 2300;
    const mph = 95;
    const omega = (rpm * 2 * Math.PI) / 60;
    const speed = mph * 0.44704;
    const s = (BALL_RADIUS_M * omega) / speed;
    const cl = liftCoefficient(s);
    expect(cl).toBeCloseTo(NATHAN_CL_A * (1 - Math.exp(-NATHAN_CL_B * s)), 8);
    range("CL", cl, 0.15, 0.28);
    range("Cd at 95 mph", dragCoefficient(speed), 0.3, 0.4);
  });

  it("thins the air with elevation and heat", () => {
    const sea = airDensityKgM3(15, 0);
    expect(sea).toBeCloseTo(1.225, 2);
    expect(airDensityKgM3(32, 0)).toBeLessThan(sea);
    expect(airDensityKgM3(15, 1600)).toBeLessThan(sea * 0.9);
  });
});

describe("spin axis round-trip", () => {
  it("recovers tilt and efficiency", () => {
    const velocity = { x: 0.4, y: 40, z: -1.2 };
    for (const tilt of [0, 15, 90, 180, 214, 270, 330]) {
      const omega = omegaFromSpin(2400, 0.8, tilt, 1, velocity);
      const back = tiltDegrees(omega, velocity);
      const diff = Math.abs((((back - tilt) % 360) + 540) % 360) - 180;
      expect(Math.abs(diff)).toBeLessThan(2);
      expect(spinEfficiency(omega, velocity)).toBeCloseTo(0.8, 2);
    }
  });

  it("formats a clock", () => {
    expect(formatClock(0, 15)).toBe("12:00");
    expect(formatClock(30, 15)).toBe("1:00");
    expect(formatClock(210, 15)).toBe("7:00");
    expect(formatClock(214, 15)).toBe("7:15");
    expect(parseClock("1:30")).toBeCloseTo(45, 5);
    expect(parseClock("12:00")).toBe(0);
  });
});

describe("textbook pitch flights", () => {
  const flights = Object.fromEntries(
    (["four-seam", "sinker", "slider", "sweeper", "curve", "changeup"] as ReferenceId[]).map((id) => [
      id,
      simulateReference(id),
    ]),
  );

  it("drops a horizontal vacuum pitch by 1/2 g t²", () => {
    const state: ReleaseState = {
      position: { x: 0, y: 2, z: 2 },
      velocity: { x: 0, y: 40, z: 0 },
      omega: { x: 0, y: 0, z: 0 },
      ssw: null,
      tempC: 15,
      elevationM: 0,
    };
    const sim = simulate(state, { flags: { drag: false, magnus: false, ssw: false }, spinDecay: false });
    const expectedZ = state.position.z - 0.5 * 9.80665 * sim.plate.t * sim.plate.t;
    expect(Math.abs(sim.plate.p.z - expectedZ)).toBeLessThan(0.02);
    expect(Math.abs(sim.plate.p.x)).toBeLessThan(0.01);
  });

  it("bleeds about 6–11 mph off a four-seamer and rides", () => {
    const metrics = flights["four-seam"].metrics;
    range("plate speed loss", metrics.releaseSpeedMph - metrics.plateSpeedMph, 6, 11);
    range("flight", metrics.flightTimeS, 0.37, 0.46);
    range("IVB", metrics.ivbIn, 14, 23);
    range("HB catcher-right", metrics.hbIn, -12, -1.5);
    expect(metrics.spinEfficiency).toBeGreaterThan(0.95);
    expect(metrics.approachAngleDeg).toBeGreaterThan(1);
  });

  it("separates sinker, sweeper, slider, and curve the way a chart does", () => {
    const fb = flights["four-seam"].metrics;
    const si = flights.sinker.metrics;
    const sl = flights.slider.metrics;
    const sw = flights.sweeper.metrics;
    const cb = flights.curve.metrics;
    const ch = flights.changeup.metrics;

    expect(si.ivbIn).toBeLessThan(fb.ivbIn - 3);
    expect(si.hbIn).toBeLessThan(fb.hbIn);

    expect(sw.hbIn).toBeGreaterThan(10);
    expect(sw.hbIn).toBeGreaterThan(sl.hbIn + 4);
    expect(Math.abs(sl.ivbIn)).toBeLessThan(10);
    expect(Math.abs(sw.ivbIn)).toBeLessThan(8);

    expect(cb.ivbIn).toBeLessThan(-5);
    expect(cb.hbIn).toBeGreaterThan(3);

    expect(ch.releaseSpeedMph).toBeLessThan(fb.releaseSpeedMph - 8);
    expect(ch.ivbIn).toBeLessThan(fb.ivbIn - 3);
    expect(ch.hbIn).toBeLessThan(fb.hbIn);
  });

  it("loses break in thinner air", () => {
    const sea = flights["four-seam"].metrics.ivbIn;
    const state = referencePitch("four-seam").state;
    const high = simulate({ ...state, elevationM: 1600 }).metrics.ivbIn;
    expect(high).toBeLessThan(sea - 1);
    expect(high).toBeGreaterThan(8);
  });

  it("gives a gyro ball much less total break than a riding four-seamer", () => {
    const fb = flights["four-seam"].metrics;
    const gyro = simulate({
      ...referencePitch("slider").state,
      omega: omegaFromSpin(2450, 0.08, 255, 1, referencePitch("slider").state.velocity),
    }).metrics;
    const fbBreak = Math.hypot(fb.ivbIn, fb.hbIn);
    const gyroBreak = Math.hypot(gyro.ivbIn, gyro.hbIn);
    expect(gyroBreak).toBeLessThan(fbBreak * 0.35);
  });
});

describe("integration sanity", () => {
  it("keeps the ball moving toward the plate", () => {
    const sim = simulateReference("four-seam");
    expect(sim.samples.length).toBeGreaterThan(20);
    expect(sim.plate.p.y).toBeCloseTo(60.5 * 0.3048, 2);
    expect(mag(sim.plate.v)).toBeGreaterThan(30);
  });
});
