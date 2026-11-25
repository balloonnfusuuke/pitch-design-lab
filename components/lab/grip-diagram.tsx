"use client";

import { GRIPS } from "@/lib/release/grips";
import type { FingerForces, GripId } from "@/lib/release/types";

function svgNum(value: number): number {
  return Math.round(value * 10000) / 10000;
}

const FINGER_LABEL: Record<keyof FingerForces, string> = {
  index: "人",
  middle: "中",
  ring: "薬",
  pinky: "小",
  thumb: "親",
};

export function GripDiagram({ grip, fingers }: { grip: GripId; fingers: FingerForces }) {
  const spec = GRIPS[grip];
  return (
    <div className="grid grid-cols-[148px_minmax(0,1fr)] items-center gap-3">
      <svg viewBox="-1.25 -1.25 2.5 2.5" className="h-[148px] w-[148px]" role="img" aria-label={`${spec.name}の握り`}>
        <circle r="1" fill="#f3efe6" stroke="#2a241c" strokeWidth="0.03" />
        <path
          d="M -0.15 -0.92 C 0.85 -0.55, 0.9 0.15, 0.15 0.72 C -0.15 0.95, -0.55 0.7, -0.35 0.35 C 0.05 -0.25, -0.15 -0.55, -0.55 -0.42 C -0.9 -0.3, -0.7 -0.85, -0.15 -0.92"
          fill="none"
          stroke="#c4372a"
          strokeWidth="0.045"
          strokeLinecap="round"
        />
        <path
          d="M 0.15 -0.92 C -0.85 -0.55, -0.9 0.15, -0.15 0.72 C 0.15 0.95, 0.55 0.7, 0.35 0.35 C -0.05 -0.25, 0.15 -0.55, 0.55 -0.42 C 0.9 -0.3, 0.7 -0.85, 0.15 -0.92"
          fill="none"
          stroke="#c4372a"
          strokeWidth="0.045"
          strokeLinecap="round"
        />
        {spec.marks.map((mark) => {
          const rad = (mark.angleDeg * Math.PI) / 180;
          const x = svgNum(Math.sin(rad) * mark.radius);
          const y = svgNum(-Math.cos(rad) * mark.radius);
          const force = fingers[mark.finger] / 100;
          const r = svgNum(0.13 + force * 0.08);
          return (
            <g key={mark.finger}>
              <circle cx={x} cy={y} r={r} fill="#d94a32" fillOpacity={svgNum(0.25 + force * 0.65)} stroke="#7a2418" strokeWidth="0.02" />
              <text x={x} y={svgNum(y + 0.045)} textAnchor="middle" fontSize="0.13" fill="#1a120f" fontFamily="sans-serif">
                {FINGER_LABEL[mark.finger]}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="text-[12px] leading-relaxed text-muted-foreground">{spec.blurb}</p>
    </div>
  );
}
