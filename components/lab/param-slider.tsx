"use client";

import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";

export function ParamSlider({
  label,
  hint,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (value: number) => void;
}) {
  return (
    <div className="grid gap-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <Label className="text-[13px] text-foreground">{label}</Label>
        <span className="lab-num text-[12px] text-[var(--gold)]">{display}</span>
      </div>
      {hint ? <p className="text-[11px] leading-snug text-muted-foreground">{hint}</p> : null}
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={(next) => onChange(next[0] ?? value)}
      />
    </div>
  );
}
