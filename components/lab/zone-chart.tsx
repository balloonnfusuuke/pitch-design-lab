"use client";

import type { PitchMetrics } from "@/lib/physics/trajectory";

export function ZoneChart({
  pitches,
  zone,
}: {
  pitches: { id: string; name: string; color: string; metrics: PitchMetrics }[];
  zone: { bottomFt: number; topFt: number };
}) {
  const width = 220;
  const height = 250;
  const n = (value: number) => Math.round(value * 100) / 100;
  const xOf = (inches: number) => n(width / 2 + inches * 4.1);
  const yOf = (feet: number) => n(height - 28 - feet * 12 * 3.15);
  const half = 17 / 2;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="捕手視点のコース">
      <rect width={width} height={height} fill="#161b13" rx="8" />
      <text x={width / 2} y="16" textAnchor="middle" fill="#a7ae99" fontSize="10">
        捕手視点　右が＋
      </text>
      <rect
        x={xOf(-half)}
        y={yOf(zone.topFt)}
        width={half * 2 * 4.1}
        height={(zone.topFt - zone.bottomFt) * 12 * 3.15}
        fill="none"
        stroke="#ebe6d6"
        strokeOpacity="0.8"
        strokeWidth="1.2"
      />
      <line x1={width / 2} y1={30} x2={width / 2} y2={height - 16} stroke="#343b2c" />
      {pitches.map((pitch) => (
        <g key={pitch.id}>
          <circle cx={xOf(pitch.metrics.plateXIn)} cy={yOf(pitch.metrics.plateZFt)} r="5.5" fill={pitch.color} stroke="#121510" strokeWidth="1" />
          <text x={xOf(pitch.metrics.plateXIn) + 8} y={yOf(pitch.metrics.plateZFt) + 3} fill={pitch.color} fontSize="9">
            {pitch.name}
          </text>
        </g>
      ))}
    </svg>
  );
}
