import { computeRelease } from "./model";
import { simulate, type IntegrateOptions, type ReleaseState, type SimulationResult } from "../physics/trajectory";
import type { ReleaseBreakdown, ReleaseInput } from "./types";

export type DesignResult = {
  input: ReleaseInput;
  breakdown: ReleaseBreakdown;
  state: ReleaseState;
  simulation: SimulationResult;
};

export function designPitch(input: ReleaseInput, options: IntegrateOptions = {}): DesignResult {
  const { state, breakdown } = computeRelease(input);
  const simulation = simulate(state, options);
  return { input, breakdown, state, simulation };
}

export const SEARCH_INTEGRATION: IntegrateOptions = {
  dt: 0.0016,
  sampleDt: 0.02,
  storeSamples: false,
};

export const INTERACTIVE_INTEGRATION: IntegrateOptions = {
  dt: 0.0005,
  sampleDt: 0.004,
  storeSamples: true,
};
