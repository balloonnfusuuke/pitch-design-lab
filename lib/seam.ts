/** Paul Bourke's baseball-seam curve, normalized onto the unit sphere. t ∈ [0, 4π]. */
export function baseballSeamPoint(t: number): [number, number, number] {
  const a = 0.4;
  const lift = Math.PI / 2 - (Math.PI / 2 - a) * Math.cos(t);
  const x = Math.sin(lift) * Math.cos(t / 2 + a * Math.sin(2 * t));
  const y = Math.sin(lift) * Math.sin(t / 2 + a * Math.sin(2 * t));
  const z = Math.cos(lift);
  const m = Math.hypot(x, y, z) || 1;
  return [x / m, y / m, z / m];
}

export function baseballSeamSamples(count = 180): [number, number, number][] {
  const points: [number, number, number][] = [];
  for (let i = 0; i < count; i++) {
    points.push(baseballSeamPoint((i / count) * Math.PI * 4));
  }
  return points;
}
