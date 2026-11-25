"use client";

import { useState } from "react";
import { ParamSlider } from "./param-slider";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { inverseDesign, type InverseResult } from "@/lib/release/inverse";
import type { ReleaseInput } from "@/lib/release/types";
import { GRIPS } from "@/lib/release/grips";

export function InversePanel({
  input,
  onApply,
}: {
  input: ReleaseInput;
  onApply: (next: ReleaseInput) => void;
}) {
  const [ivb, setIvb] = useState(2);
  const [hb, setHb] = useState(15);
  const [velo, setVelo] = useState(82);
  const [lockGrip, setLockGrip] = useState(false);
  const [searching, setSearching] = useState(false);
  const [result, setResult] = useState<InverseResult | null>(null);

  const run = () => {
    setSearching(true);
    setResult(null);
    window.setTimeout(() => {
      const found = inverseDesign({ ivbIn: ivb, hbIn: hb, veloMph: velo }, input, { lockGrip });
      setResult(found);
      setSearching(false);
    }, 30);
  };

  return (
    <div className="grid gap-4">
      <p className="text-[12px] leading-relaxed text-muted-foreground">
        欲しい変化を入れると、今の手の長さの範囲で握り・指・腕を探します。横変化は捕手から見て右が＋です。右投手のスイーパーはプラス、シンカーはマイナス。
      </p>
      <ParamSlider label="目標の縦変化" value={ivb} min={-25} max={28} step={0.5} display={`${ivb > 0 ? "+" : ""}${ivb.toFixed(1)} in`} onChange={setIvb} />
      <ParamSlider label="目標の横変化" value={hb} min={-25} max={25} step={0.5} display={`${hb > 0 ? "+" : ""}${hb.toFixed(1)} in`} onChange={setHb} />
      <ParamSlider label="目標の球速" value={velo} min={68} max={102} step={0.5} display={`${velo.toFixed(1)} mph`} onChange={setVelo} />
      <label className="flex items-center justify-between gap-3 text-[13px]">
        <span>握りは{GRIPS[input.grip].name}のまま探す</span>
        <Switch checked={lockGrip} onCheckedChange={setLockGrip} />
      </label>
      <Button type="button" onClick={run} disabled={searching}>
        {searching ? "探索しています…" : "この変化を探す"}
      </Button>
      {result ? (
        <div className="grid gap-3 rounded-lg border border-border bg-card/60 p-3">
          <p className="text-[13px] leading-relaxed">{result.summary}</p>
          <p className="lab-num text-[12px] text-[var(--gold)]">
            結果 {result.metrics.releaseSpeedMph.toFixed(1)} mph / {result.metrics.tiltClock} / IVB{" "}
            {result.metrics.ivbIn.toFixed(1)} / HB {result.metrics.hbIn.toFixed(1)}
          </p>
          {result.bottlenecks.length > 0 ? (
            <ul className="grid gap-2">
              {result.bottlenecks.map((item) => (
                <li key={item.parameter} className="border-l-2 border-[#d94a32] pl-3 text-[13px] leading-relaxed">
                  {item.text}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-[13px] text-muted-foreground">可動域の端には当たっていません。</p>
          )}
          <Button type="button" variant="outline" onClick={() => onApply(result.input)}>
            この解をスライダーに入れる
          </Button>
        </div>
      ) : null}
    </div>
  );
}
