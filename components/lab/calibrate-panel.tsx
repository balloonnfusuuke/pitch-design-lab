"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { calibrateMeasured, parsePitchCsv, releaseFromMeasured, type MeasuredPitch } from "@/lib/calibration/csv";
import { simulate, type PitchMetrics, type ReleaseState, type Sample } from "@/lib/physics/trajectory";
import type { Vec3 } from "@/lib/physics/vec";
import type { ReleaseInput } from "@/lib/release/types";

export function CalibratePanel({
  input,
  metrics,
  release,
  onAdd,
}: {
  input: ReleaseInput;
  metrics: PitchMetrics;
  release: ReleaseState;
  onAdd: (pitch: { name: string; metrics: PitchMetrics; samples: Sample[]; omega: Vec3 }) => void;
}) {
  const [rows, setRows] = useState<MeasuredPitch[]>([]);
  const [selected, setSelected] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const measured = rows[selected];
  const comparison = measured
    ? calibrateMeasured(
        measured,
        metrics,
        release.position,
        { x: input.aimXIn * 0.0254, z: input.aimZFt * 0.3048 },
        { tempC: input.tempC, elevationM: input.elevationM },
      )
    : null;

  return (
    <div className="grid gap-4">
      <p className="text-[12px] leading-relaxed text-muted-foreground">
        Rapsodo や TrackMan 風の CSV（球速、回転数、ティルト、縦横の変化）を読み、今の設計との差を出します。ティルトがあれば、その回転で飛ばした軌道を比較に重ねられます。
      </p>
      <a className="text-[12px] text-[var(--gold)] underline-offset-2 hover:underline" href="/samples/measured-pitches.csv" download>
        サンプル CSV をダウンロード
      </a>
      <input
        type="file"
        accept=".csv,text/csv"
        className="text-[12px]"
        onChange={async (event) => {
          const file = event.target.files?.[0];
          if (!file) return;
          const parsed = parsePitchCsv(await file.text());
          if (parsed.length === 0) {
            setError("球速と回転数の列が読めませんでした。1 行目は name, velo_mph, spin_rpm, tilt, ivb_in, hb_in のようにしてください。");
            setRows([]);
            return;
          }
          setError(null);
          setRows(parsed);
          setSelected(0);
        }}
      />
      {error ? <p className="text-[12px] text-[#d94a32]">{error}</p> : null}
      {rows.length > 0 ? (
        <div className="flex flex-wrap gap-1">
          {rows.map((row, index) => (
            <button
              key={`${row.name}-${index}`}
              type="button"
              onClick={() => setSelected(index)}
              className={`rounded-md border px-2 py-1 text-[12px] ${index === selected ? "border-[var(--gold)] bg-accent" : "border-border"}`}
            >
              {row.name}
            </button>
          ))}
        </div>
      ) : null}
      {comparison && measured ? (
        <div className="grid gap-3">
          {comparison.note ? <p className="text-[12px] text-[var(--gold)]">{comparison.note}</p> : null}
          <table className="w-full text-left text-[12px]">
            <thead className="text-muted-foreground">
              <tr>
                <th className="py-1 font-normal">項目</th>
                <th className="py-1 font-normal">実測</th>
                <th className="py-1 font-normal">モデル</th>
                <th className="py-1 font-normal">差</th>
              </tr>
            </thead>
            <tbody className="lab-num">
              <tr>
                <td className="py-1">球速</td>
                <td>{measured.veloMph.toFixed(1)}</td>
                <td>{metrics.releaseSpeedMph.toFixed(1)}</td>
                <td>{fmtDelta(comparison.deltas.veloMph)}</td>
              </tr>
              <tr>
                <td className="py-1">回転</td>
                <td>{Math.round(measured.spinRpm)}</td>
                <td>{Math.round(metrics.spinRpm)}</td>
                <td>{fmtDelta(comparison.deltas.spinRpm, 0)}</td>
              </tr>
              <tr>
                <td className="py-1">IVB</td>
                <td>{measured.ivbIn?.toFixed(1) ?? "—"}</td>
                <td>{metrics.ivbIn.toFixed(1)}</td>
                <td>{fmtDelta(comparison.deltas.ivbIn)}</td>
              </tr>
              <tr>
                <td className="py-1">横</td>
                <td>{measured.hbIn?.toFixed(1) ?? "—"}</td>
                <td>{metrics.hbIn.toFixed(1)}</td>
                <td>{fmtDelta(comparison.deltas.hbIn)}</td>
              </tr>
            </tbody>
          </table>
          <ul className="grid gap-2">
            {comparison.suggestions.map((hint) => (
              <li key={hint.id} className="border-l-2 border-[#e0b15a] pl-3 text-[13px] leading-relaxed">
                {hint.text}
              </li>
            ))}
          </ul>
          <Button
            type="button"
            variant="outline"
            onClick={() => {
              const state = releaseFromMeasured(
                measured,
                release.position,
                { x: input.aimXIn * 0.0254, z: input.aimZFt * 0.3048 },
                { tempC: input.tempC, elevationM: input.elevationM },
              );
              if (!state) return;
              const flown = simulate(state);
              onAdd({
                name: measured.name,
                metrics: flown.metrics,
                samples: flown.samples,
                omega: state.omega,
              });
            }}
          >
            実測の回転で飛ばして比較に入れる
          </Button>
        </div>
      ) : null}
    </div>
  );
}

function fmtDelta(value: number | null, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const text = value.toFixed(digits);
  return value > 0 ? `+${text}` : text;
}
