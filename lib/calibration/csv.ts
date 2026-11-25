import { MPH_TO_MS, RUBBER_TO_PLATE_M } from "../physics/constants";
import { omegaFromSpin, parseClock } from "../physics/spin";
import {
  simulate,
  velocityTowardTarget,
  type PitchMetrics,
  type ReleaseState,
  type SimulationResult,
} from "../physics/trajectory";
import type { Hint } from "../release/hints";

export type MeasuredPitch = {
  name: string;
  veloMph: number;
  spinRpm: number;
  tiltDeg: number | null;
  spinEfficiency: number | null;
  ivbIn: number | null;
  hbIn: number | null;
  extensionFt: number | null;
};

export type CalibrationRow = {
  measured: MeasuredPitch;
  simulation: SimulationResult | null;
  deltas: {
    veloMph: number | null;
    spinRpm: number | null;
    ivbIn: number | null;
    hbIn: number | null;
  };
  suggestions: Hint[];
  note: string | null;
};

const HEADER_MAP: Record<string, keyof MeasuredPitch | "ignore"> = {
  name: "name",
  pitch: "name",
  球種: "name",
  velo: "veloMph",
  velo_mph: "veloMph",
  velocity: "veloMph",
  release_speed: "veloMph",
  speed: "veloMph",
  球速: "veloMph",
  spin: "spinRpm",
  spin_rpm: "spinRpm",
  spinrate: "spinRpm",
  spin_rate: "spinRpm",
  回転数: "spinRpm",
  tilt: "tiltDeg",
  spin_direction: "tiltDeg",
  spin_axis: "tiltDeg",
  axis: "tiltDeg",
  ティルト: "tiltDeg",
  spin_efficiency: "spinEfficiency",
  efficiency: "spinEfficiency",
  効率: "spinEfficiency",
  ivb: "ivbIn",
  ivb_in: "ivbIn",
  induced_vert_break: "ivbIn",
  vertical_break: "ivbIn",
  pfx_z: "ivbIn",
  縦変化: "ivbIn",
  hb: "hbIn",
  hb_in: "hbIn",
  horz_break: "hbIn",
  horizontal_break: "hbIn",
  pfx_x: "hbIn",
  横変化: "hbIn",
  extension: "extensionFt",
  extension_ft: "extensionFt",
  エクステンション: "extensionFt",
};

function normalizeHeader(header: string): string {
  return header.trim().toLowerCase().replace(/\s+/g, "_");
}

export function parsePitchCsv(text: string): MeasuredPitch[] {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith("#"));
  if (lines.length < 2) return [];
  const headers = splitCsvLine(lines[0]).map(normalizeHeader);
  const columns = headers.map((header) => {
    if (header in HEADER_MAP) return HEADER_MAP[header];
    if (header.includes("pfx") && header.includes("x")) return "hbIn" as const;
    if (header.includes("pfx") && (header.includes("z") || header.includes("vert"))) return "ivbIn" as const;
    return "ignore" as const;
  });
  const feetColumns = new Set(
    headers.map((header, index) => (header.startsWith("pfx") ? index : -1)).filter((index) => index >= 0),
  );

  const pitches: MeasuredPitch[] = [];
  for (const line of lines.slice(1)) {
    const cells = splitCsvLine(line);
    const draft: Partial<MeasuredPitch> = { name: "実測" };
    cells.forEach((cell, index) => {
      const key = columns[index];
      if (!key || key === "ignore") return;
      if (key === "name") {
        draft.name = cell || "実測";
        return;
      }
      const numeric = Number(cell);
      if (key === "tiltDeg") {
        draft.tiltDeg = parseClock(cell) ?? (Number.isFinite(numeric) ? numeric : null);
        return;
      }
      if (!Number.isFinite(numeric)) return;
      let value = numeric;
      if (feetColumns.has(index) && (key === "ivbIn" || key === "hbIn")) value *= 12;
      draft[key] = value;
    });
    if (draft.veloMph == null || draft.spinRpm == null) continue;
    pitches.push({
      name: draft.name ?? "実測",
      veloMph: draft.veloMph,
      spinRpm: draft.spinRpm,
      tiltDeg: draft.tiltDeg ?? null,
      spinEfficiency: draft.spinEfficiency ?? null,
      ivbIn: draft.ivbIn ?? null,
      hbIn: draft.hbIn ?? null,
      extensionFt: draft.extensionFt ?? null,
    });
  }
  return pitches;
}

function splitCsvLine(line: string): string[] {
  const cells: string[] = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      quoted = !quoted;
      continue;
    }
    if (char === "," && !quoted) {
      cells.push(current.trim());
      current = "";
      continue;
    }
    current += char;
  }
  cells.push(current.trim());
  return cells;
}

export function releaseFromMeasured(
  measured: MeasuredPitch,
  releasePoint: { x: number; y: number; z: number },
  aim: { x: number; z: number },
  environment: { tempC: number; elevationM: number },
): ReleaseState | null {
  if (measured.tiltDeg == null) return null;
  const position = {
    x: releasePoint.x,
    y: measured.extensionFt != null ? measured.extensionFt * 0.3048 : releasePoint.y,
    z: releasePoint.z,
  };
  const velocity = velocityTowardTarget(
    position,
    { x: aim.x, y: RUBBER_TO_PLATE_M, z: aim.z },
    measured.veloMph * MPH_TO_MS,
  );
  const efficiency = measured.spinEfficiency ?? 0.85;
  const omega = omegaFromSpin(measured.spinRpm, efficiency, measured.tiltDeg, 1, velocity);
  return {
    position,
    velocity,
    omega,
    ssw: null,
    tempC: environment.tempC,
    elevationM: environment.elevationM,
  };
}

export function calibrateMeasured(
  measured: MeasuredPitch,
  designed: PitchMetrics,
  releasePoint: { x: number; y: number; z: number },
  aim: { x: number; z: number },
  environment: { tempC: number; elevationM: number },
): CalibrationRow {
  const state = releaseFromMeasured(measured, releasePoint, aim, environment);
  const simulation = state ? simulate(state) : null;
  const suggestions: Hint[] = [];
  const note =
    measured.tiltDeg == null
      ? "ティルト（時計か度数）が無いので、実測値の軌道は飛ばせません。差分のコメントだけ出します。"
      : measured.spinEfficiency == null
        ? "回転効率が空なので 85% と仮定して軌道を描いています。"
        : null;

  if (measured.ivbIn != null) {
    const gap = measured.ivbIn - designed.ivbIn;
    if (gap > 2) {
      suggestions.push({
        id: "cal-ivb-low",
        severity: "high",
        text: `モデルの縦変化が実測より ${gap.toFixed(1)} in 小さいです。バックスピン（人差し指・中指、回転効率）がまだ足りません。`,
      });
    } else if (gap < -2) {
      suggestions.push({
        id: "cal-ivb-high",
        severity: "mid",
        text: `モデルの縦変化が実測より ${Math.abs(gap).toFixed(1)} in 大きいです。回内で軸を倒すか、回転数を下げると実測に寄ります。`,
      });
    }
  }
  if (measured.hbIn != null) {
    const gap = measured.hbIn - designed.hbIn;
    if (Math.abs(gap) > 2) {
      const toward = gap > 0 ? "捕手の右側" : "捕手の左側";
      suggestions.push({
        id: "cal-hb",
        severity: "high",
        text: `横変化が実測より ${toward} に ${Math.abs(gap).toFixed(1)} in 足りません。回内・回外と、最後に離れる指をその方向へ寄せてください。`,
      });
    }
  }
  if (measured.veloMph - designed.releaseSpeedMph > 2) {
    suggestions.push({
      id: "cal-velo-low",
      severity: "high",
      text: `球速が実測より ${(measured.veloMph - designed.releaseSpeedMph).toFixed(1)} mph 遅いです。腕の速度か、手首の伸展が不足しています。`,
    });
  } else if (designed.releaseSpeedMph - measured.veloMph > 2) {
    suggestions.push({
      id: "cal-velo-high",
      severity: "mid",
      text: `球速が実測より ${(designed.releaseSpeedMph - measured.veloMph).toFixed(1)} mph 速いです。腕の速度を下げるか、チェンジアップのように指で殺します。`,
    });
  }
  if (measured.spinRpm - designed.spinRpm > 150) {
    suggestions.push({
      id: "cal-spin-low",
      severity: "mid",
      text: `回転数が実測より約 ${Math.round(measured.spinRpm - designed.spinRpm)} rpm 低いです。接線方向の指の力か、ボールに触れている時間を増やします。`,
    });
  }
  if (suggestions.length === 0) {
    suggestions.push({
      id: "cal-close",
      severity: "info",
      text: "球速・回転・変化は、この実測とかなり近いです。残差は縫い目の向きか、当日の空気密度の範囲です。",
    });
  }

  return {
    measured,
    simulation,
    deltas: {
      veloMph: designed.releaseSpeedMph - measured.veloMph,
      spinRpm: designed.spinRpm - measured.spinRpm,
      ivbIn: measured.ivbIn == null ? null : designed.ivbIn - measured.ivbIn,
      hbIn: measured.hbIn == null ? null : designed.hbIn - measured.hbIn,
    },
    suggestions,
    note,
  };
}
