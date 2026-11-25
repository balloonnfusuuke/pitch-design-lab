"use client";

import dynamic from "next/dynamic";
import { Component, useEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from "react";
import { DesignControls } from "./controls";
import { CalibratePanel } from "./calibrate-panel";
import { HandPanel } from "./hand-panel";
import { InversePanel } from "./inverse-panel";
import { MetricsPanel, type ListedPitch } from "./metrics-panel";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { CameraName, ScenePitch } from "./field-scene";
import { GRIPS } from "@/lib/release/grips";
import { designPitch } from "@/lib/release/design";
import { trainingHints } from "@/lib/release/hints";
import { presetInput } from "@/lib/release/grips";
import { referencePitch, type ReferenceId } from "@/lib/physics/reference";
import { simulate, type Sample } from "@/lib/physics/trajectory";
import type { Vec3 } from "@/lib/physics/vec";
import type { ReleaseInput } from "@/lib/release/types";

const FieldScene = dynamic(() => import("./field-scene").then((mod) => mod.FieldScene), {
  ssr: false,
  loading: () => (
    <div className="grid h-full min-h-[420px] place-items-center text-sm text-muted-foreground">球場を準備しています…</div>
  ),
});

const COMPARE_COLORS = ["#7eb8c9", "#e0b15a", "#b089e0", "#8fbf78"];

type SavedPitch = {
  id: string;
  name: string;
  color: string;
  samples: Sample[];
  metrics: ListedPitch["metrics"];
  omega: Vec3;
  handedness: ReleaseInput["handedness"];
};

const CAMERAS: { id: CameraName; label: string }[] = [
  { id: "diamond", label: "斜め" },
  { id: "catcher", label: "捕手" },
  { id: "pitcher", label: "投手" },
  { id: "side", label: "横" },
  { id: "above", label: "上" },
];

export function LabApp() {
  const [input, setInput] = useState<ReleaseInput>(() => presetInput("four-seam"));
  const [saved, setSaved] = useState<SavedPitch[]>([]);
  const [showGhost, setShowGhost] = useState(true);
  const [enlarge, setEnlarge] = useState(true);
  const [zone, setZone] = useState({ bottomFt: 1.5, topFt: 3.5 });
  const [camera, setCamera] = useState<CameraName>("diamond");
  const [playing, setPlaying] = useState(true);
  const [rate, setRate] = useState(0.4);
  const timeRef = useRef(0);
  const playingRef = useRef(true);
  const rateRef = useRef(0.4);
  const scrubber = useRef<HTMLInputElement>(null);
  const timeLabel = useRef<HTMLSpanElement>(null);

  const result = useMemo(() => designPitch(input), [input]);
  const hints = useMemo(() => trainingHints(input, result.simulation.metrics), [input, result]);
  const grip = GRIPS[input.grip];

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);
  useEffect(() => {
    rateRef.current = rate;
  }, [rate]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.code !== "Space") return;
      const tag = (event.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || tag === "BUTTON") return;
      event.preventDefault();
      setPlaying((value) => !value);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => {
    let frame = 0;
    const tick = () => {
      const time = timeRef.current ?? 0;
      if (scrubber.current && document.activeElement !== scrubber.current) {
        scrubber.current.value = String(time);
      }
      if (timeLabel.current) timeLabel.current.textContent = `${time.toFixed(2)} s`;
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  const duration = Math.max(result.simulation.metrics.flightTimeS, ...saved.map((pitch) => pitch.metrics.flightTimeS), 0.2);

  const scenePitches: ScenePitch[] = [
    {
      id: "active",
      name: grip.name,
      color: grip.color,
      samples: result.simulation.samples,
      spinless: result.simulation.spinlessSamples,
      omega: result.state.omega,
      active: true,
    },
    ...saved.map((pitch) => ({
      id: pitch.id,
      name: pitch.name,
      color: pitch.color,
      samples: pitch.samples,
      spinless: [] as Sample[],
      omega: pitch.omega,
      active: false,
    })),
  ];

  const addSaved = (pitch: Omit<SavedPitch, "id" | "color" | "handedness"> & { color?: string; handedness?: ReleaseInput["handedness"] }) => {
    setSaved((prev) => {
      const next: SavedPitch = {
        id: crypto.randomUUID(),
        name: pitch.name,
        color: pitch.color ?? COMPARE_COLORS[prev.length % COMPARE_COLORS.length],
        samples: pitch.samples,
        metrics: pitch.metrics,
        omega: pitch.omega,
        handedness: pitch.handedness ?? input.handedness,
      };
      return [...prev, next].slice(-4);
    });
  };

  const addReferences = () => {
    const ids: ReferenceId[] = ["four-seam", "sweeper", "curve"];
    const refs: SavedPitch[] = ids.map((id, index) => {
      const ref = referencePitch(id);
      const flown = simulate(ref.state);
      return {
        id: `ref-${id}`,
        name: ref.name,
        color: COMPARE_COLORS[index],
        samples: flown.samples,
        metrics: flown.metrics,
        omega: ref.state.omega,
        handedness: "R",
      };
    });
    setSaved(refs);
  };

  return (
    <div className="flex min-h-svh flex-col lg:h-svh">
      <header className="flex flex-wrap items-end justify-between gap-3 border-b border-border px-4 py-3 lg:px-5">
        <div>
          <p className="text-[11px] tracking-[0.18em] text-[var(--gold)]">PITCH DESIGN LAB</p>
          <h1 className="lab-display text-2xl leading-tight">ピッチデザイン・ラボ</h1>
          <p className="max-w-xl text-[12px] text-muted-foreground">
            握りと指、腕の抜き方から球を設計し、本塁までの軌道を見る。
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="outline" size="sm" onClick={addReferences}>
            典型球を重ねる
          </Button>
          <Button
            type="button"
            size="sm"
            onClick={() =>
              addSaved({
                name: grip.name,
                samples: result.simulation.samples,
                metrics: result.simulation.metrics,
                omega: result.state.omega,
                color: grip.color,
              })
            }
          >
            比較に追加
          </Button>
        </div>
      </header>

      <div className="grid flex-1 lg:min-h-0 lg:grid-cols-[340px_minmax(0,1fr)_320px]">
        <aside className="border-b border-border lg:min-h-0 lg:overflow-hidden lg:border-r lg:border-b-0">
          <Tabs defaultValue="design" className="lg:flex lg:h-full lg:min-h-0 lg:flex-col">
            <TabsList className="mx-3 mt-3 grid w-auto grid-cols-5">
              <TabsTrigger value="design">握り</TabsTrigger>
              <TabsTrigger value="search">探索</TabsTrigger>
              <TabsTrigger value="hand">手</TabsTrigger>
              <TabsTrigger value="cal">実測</TabsTrigger>
              <TabsTrigger value="note">根拠</TabsTrigger>
            </TabsList>
            <div className="lab-scroll px-3 py-4 lg:min-h-0 lg:flex-1 lg:overflow-y-auto">
              <TabsContent value="design">
                <DesignControls input={input} onChange={setInput} zone={zone} onZone={setZone} />
              </TabsContent>
              <TabsContent value="search">
                <InversePanel input={input} onApply={setInput} />
              </TabsContent>
              <TabsContent value="hand">
                <HandPanel hand={input.hand} onChange={(hand) => setInput({ ...input, hand })} />
              </TabsContent>
              <TabsContent value="cal">
                <CalibratePanel
                  input={input}
                  metrics={result.simulation.metrics}
                  release={result.state}
                  onAdd={(pitch) => addSaved(pitch)}
                />
              </TabsContent>
              <TabsContent value="note">
                <PhysicsNote />
              </TabsContent>
            </div>
          </Tabs>
        </aside>

        <section className="flex min-h-[460px] flex-col lg:min-h-0">
          <div className="relative min-h-[420px] flex-1">
            <SceneBoundary>
              <FieldScene
                pitches={scenePitches}
                showGhost={showGhost}
                enlargeBall={enlarge}
                zone={zone}
                camera={camera}
                timeRef={timeRef as RefObject<number>}
                playingRef={playingRef as RefObject<boolean>}
                rateRef={rateRef as RefObject<number>}
                duration={duration}
              />
            </SceneBoundary>
            <div className="pointer-events-none absolute top-3 left-3 text-[11px] text-[#c9c3b0]">
              金色の線は判断点（本塁の手前 6.4 m）。白い枠がストライクゾーン。
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 border-t border-border px-3 py-2">
            <Button type="button" size="sm" variant="outline" onClick={() => setPlaying((value) => !value)}>
              {playing ? "停止" : "再生"}
            </Button>
            <input
              ref={scrubber}
              aria-label="再生位置"
              type="range"
              min={0}
              max={duration}
              step={0.001}
              defaultValue={0}
              className="min-w-32 flex-1 accent-[#d94a32]"
              onPointerDown={() => {
                playingRef.current = false;
                setPlaying(false);
              }}
              onInput={(event) => {
                timeRef.current = Number((event.target as HTMLInputElement).value);
              }}
            />
            <span ref={timeLabel} className="lab-num w-14 text-[12px] text-muted-foreground">
              0.00 s
            </span>
            <label className="flex items-center gap-1 text-[12px] text-muted-foreground">
              速度
              <select
                className="rounded-md border border-border bg-card px-1 py-1"
                value={rate}
                onChange={(event) => setRate(Number(event.target.value))}
              >
                <option value={0.25}>0.25×</option>
                <option value={0.4}>0.4×</option>
                <option value={1}>1×</option>
              </select>
            </label>
            <div className="flex gap-1">
              {CAMERAS.map((shot) => (
                <button
                  key={shot.id}
                  type="button"
                  onClick={() => setCamera(shot.id)}
                  className={`rounded-md border px-2 py-1 text-[12px] ${camera === shot.id ? "border-[var(--gold)]" : "border-border"}`}
                >
                  {shot.label}
                </button>
              ))}
            </div>
            <button type="button" className="text-[12px] text-muted-foreground" onClick={() => setShowGhost((value) => !value)}>
              {showGhost ? "無回転の線を隠す" : "無回転の線を出す"}
            </button>
            <button type="button" className="text-[12px] text-muted-foreground" onClick={() => setEnlarge((value) => !value)}>
              {enlarge ? "ボールを実寸に" : "ボールを拡大"}
            </button>
          </div>
        </section>

        <aside className="lab-scroll border-t border-border px-4 py-4 lg:min-h-0 lg:overflow-y-auto lg:border-t-0 lg:border-l">
          <MetricsPanel
            name={grip.name}
            color={grip.color}
            metrics={result.simulation.metrics}
            breakdown={result.breakdown}
            hints={hints}
            hand={input.handedness}
            zone={zone}
            activeSamples={result.simulation.samples}
            saved={saved}
            onRemove={(id) => setSaved((prev) => prev.filter((pitch) => pitch.id !== id))}
          />
        </aside>
      </div>
    </div>
  );
}

function PhysicsNote() {
  return (
    <div className="grid gap-3 text-[13px] leading-relaxed text-muted-foreground">
      <p className="text-foreground">軌道は、公開されている野球の空力に沿っています。</p>
      <p>
        揚力係数は Alan Nathan の 2020 年の式、CL = 0.336 (1 − e^(−6.041 S)) です。S はボール半径 × 進行方向に垂直な回転 ÷ 速さです。抗力は投球速度帯で測られてきた 0.30–0.40 のあたりを通る滑らかな曲線で、単一の確定式ではありません。
      </p>
      <p>
        リリースは、指の接線力 × 接触時間 ÷ 慣性モーメント、という角力積です。力の上限と接触時間は、プリセットが MLB でよく見る球速・回転・変化に入るように合わせています。腕の角度と回内は、スピン軸が前腕とほぼ直交するという投球の観察に沿ったルールです。
      </p>
      <p>
        縫い目ウェイクは、向きと強さを握りごとに置いた付加項です。縫い目の流体計算ではありません。スライダーを 0 にすると、その分の変化は消えます。
      </p>
      <p>判断点の金色の線は本塁の 6.4 m 手前です。打者が振りに行く頃に、球同士がまだ重なっているかを見るための印です。</p>
    </div>
  );
}

class SceneBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }
  render() {
    if (this.state.failed) {
      return (
        <div className="grid h-full min-h-[420px] place-items-center px-6 text-center text-sm text-muted-foreground">
          WebGL を初期化できませんでした。数値、コース図、トレーニングの文章はそのまま使えます。
        </div>
      );
    }
    return this.props.children;
  }
}
