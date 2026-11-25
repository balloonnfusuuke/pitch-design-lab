"use client";

import { ZoneChart } from "./zone-chart";
import { Badge } from "@/components/ui/badge";
import { breakWords, inchesAndCm, mphAndKmh, plateCall } from "@/lib/format";
import { RUBBER_TO_PLATE_M } from "@/lib/physics/constants";
import { separationInches, tunnelPlaneY, type PitchMetrics } from "@/lib/physics/trajectory";
import type { Sample } from "@/lib/physics/trajectory";
import type { Hint } from "@/lib/release/hints";
import type { ReleaseBreakdown } from "@/lib/release/types";
import type { Handedness } from "@/lib/release/types";

export type ListedPitch = {
  id: string;
  name: string;
  color: string;
  metrics: PitchMetrics;
  samples: Sample[];
  handedness: Handedness;
};

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card/70 px-3 py-2">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="lab-num text-[22px] leading-none text-[var(--gold)]">{value}</div>
      {sub ? <div className="mt-1 text-[11px] text-muted-foreground">{sub}</div> : null}
    </div>
  );
}

export function MetricsPanel({
  name,
  color,
  metrics,
  breakdown,
  hints,
  hand,
  zone,
  activeSamples,
  saved,
  onRemove,
}: {
  name: string;
  color: string;
  metrics: PitchMetrics;
  breakdown: ReleaseBreakdown;
  hints: Hint[];
  hand: Handedness;
  zone: { bottomFt: number; topFt: number };
  activeSamples: Sample[];
  saved: ListedPitch[];
  onRemove: (id: string) => void;
}) {
  const velo = mphAndKmh(metrics.releaseSpeedMph);
  const plate = mphAndKmh(metrics.plateSpeedMph);
  const ivb = inchesAndCm(metrics.ivbIn);
  const hb = inchesAndCm(metrics.hbIn);
  const call = plateCall(metrics, zone);
  const chart = [
    { id: "active", name, color, metrics },
    ...saved.map((pitch) => ({ id: pitch.id, name: pitch.name, color: pitch.color, metrics: pitch.metrics })),
  ];

  return (
    <div className="grid gap-4">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
          <h2 className="lab-display text-lg">{name}</h2>
        </div>
        <Badge variant={call === "strike" ? "default" : "outline"}>{call === "strike" ? "ストライク" : "ボール"}</Badge>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <Stat label="リリース球速" value={`${velo.mph}`} sub={`${velo.kmh} km/h　本塁 ${plate.mph} mph`} />
        <Stat label="回転数" value={`${Math.round(metrics.spinRpm)}`} sub={`効率 ${Math.round(metrics.spinEfficiency * 100)}%`} />
        <Stat label="縦の変化 IVB" value={ivb.inches} sub={`${ivb.cm} cm　上向きが＋`} />
        <Stat label="横の変化" value={hb.inches} sub={`${hb.cm} cm　${breakWords(metrics, hand)}`} />
      </div>

      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-[12px]">
        <dt className="text-muted-foreground">スピン軸</dt>
        <dd className="lab-num text-right">{metrics.tiltClock}（{metrics.tiltClockExact}）</dd>
        <dt className="text-muted-foreground">コース</dt>
        <dd className="lab-num text-right">
          横 {metrics.plateXIn > 0 ? "+" : ""}
          {metrics.plateXIn.toFixed(1)} in / 高さ {metrics.plateZFt.toFixed(2)} ft
        </dd>
        <dt className="text-muted-foreground">落下（実距離）</dt>
        <dd className="lab-num text-right">{metrics.totalDropIn.toFixed(0)} in</dd>
        <dt className="text-muted-foreground">入射角</dt>
        <dd className="lab-num text-right">{metrics.approachAngleDeg.toFixed(1)}°</dd>
        <dt className="text-muted-foreground">飛行時間</dt>
        <dd className="lab-num text-right">{metrics.flightTimeS.toFixed(3)} s</dd>
      </dl>

      <ZoneChart pitches={chart} zone={zone} />

      <section className="grid gap-2">
        <h3 className="text-[12px] tracking-wide text-muted-foreground">トレーニング</h3>
        <ul className="grid gap-2">
          {hints.map((hint) => (
            <li
              key={hint.id}
              className={`border-l-2 pl-3 text-[13px] leading-relaxed ${
                hint.severity === "high"
                  ? "border-[#d94a32]"
                  : hint.severity === "mid"
                    ? "border-[#e0b15a]"
                    : "border-[#8fbf78]"
              }`}
            >
              {hint.text}
            </li>
          ))}
        </ul>
      </section>

      {saved.length > 0 ? (
        <section className="grid gap-2">
          <h3 className="text-[12px] tracking-wide text-muted-foreground">比較との間隔</h3>
          <ul className="grid gap-2">
            {saved.map((pitch) => (
              <CompareRow
                key={pitch.id}
                pitch={pitch}
                onRemove={onRemove}
                tunnelIn={separationInches(activeSamples, pitch.samples, tunnelPlaneY())}
                plateIn={separationInches(activeSamples, pitch.samples, RUBBER_TO_PLATE_M)}
              />
            ))}
          </ul>
        </section>
      ) : null}

      <details className="rounded-lg border border-border px-3 py-2 text-[12px] leading-relaxed text-muted-foreground">
        <summary className="cursor-pointer text-foreground">モデルの内訳</summary>
        <p className="mt-2">
          接線力 {breakdown.tangentialForceN.toFixed(0)} N × 接触 {breakdown.contactMs.toFixed(1)} ms で回転{" "}
          {Math.round(breakdown.spinRpm)} rpm。手先 {breakdown.handSpeedMph.toFixed(1)} mph、腕の角速度{" "}
          {breakdown.armAngularVelRadS.toFixed(1)} rad/s。リリース高 {breakdown.releaseZFt.toFixed(2)} ft、横{" "}
          {breakdown.releaseXFt.toFixed(2)} ft。中指レバーは基準の {breakdown.middleLeverScale.toFixed(2)} 倍。縫い目項 C ={" "}
          {breakdown.sswCoeff.toFixed(3)}。
        </p>
        <p className="mt-2">
          時計は投手から見て 12 時がバックスピン、3 時が投手の右手側です。灰色の線は回転も縫い目もない場合の軌道で、そこからの差が IVB / 横変化です。
        </p>
      </details>
    </div>
  );
}

function CompareRow({
  pitch,
  onRemove,
  tunnelIn,
  plateIn,
}: {
  pitch: ListedPitch;
  onRemove: (id: string) => void;
  tunnelIn: number | null;
  plateIn: number | null;
}) {
  return (
    <li className="rounded-md border border-border px-2 py-1.5">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-2 text-[13px]">
          <span className="h-2 w-2 rounded-full" style={{ background: pitch.color }} />
          {pitch.name}
        </span>
        <button type="button" className="text-[11px] text-muted-foreground hover:text-foreground" onClick={() => onRemove(pitch.id)}>
          外す
        </button>
      </div>
      <p className="mt-1 text-[11px] text-muted-foreground">
        判断点 {tunnelIn == null ? "—" : `${(tunnelIn * 2.54).toFixed(0)} cm`}　本塁{" "}
        {plateIn == null ? "—" : `${(plateIn * 2.54).toFixed(0)} cm`}
        {tunnelIn != null && plateIn != null && tunnelIn < 6 && plateIn > 10 ? "　途中まで重なっています。" : ""}
      </p>
    </li>
  );
}

export function separations(
  active: Sample[],
  saved: ListedPitch[],
): { id: string; tunnelIn: number | null; plateIn: number | null }[] {
  return saved.map((pitch) => ({
    id: pitch.id,
    tunnelIn: separationInches(active, pitch.samples, tunnelPlaneY()),
    plateIn: separationInches(active, pitch.samples, RUBBER_TO_PLATE_M),
  }));
}
