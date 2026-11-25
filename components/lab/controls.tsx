"use client";

import { GripDiagram } from "./grip-diagram";
import { ParamSlider } from "./param-slider";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { GRIP_LIST, presetInput } from "@/lib/release/grips";
import type { FingerId, GripId, LastContact, ReleaseInput } from "@/lib/release/types";

const FINGERS: { id: FingerId; label: string }[] = [
  { id: "index", label: "人差し指" },
  { id: "middle", label: "中指" },
  { id: "ring", label: "薬指" },
  { id: "pinky", label: "小指" },
  { id: "thumb", label: "親指" },
];

const CONTACTS: { id: LastContact; label: string }[] = [
  { id: "even", label: "同時" },
  { id: "index", label: "人差" },
  { id: "middle", label: "中" },
  { id: "ring", label: "薬" },
  { id: "pinky", label: "小" },
  { id: "thumb", label: "親" },
];

function slotWord(deg: number): string {
  if (deg >= 75) return "オーバー";
  if (deg >= 55) return "スリークォーター";
  if (deg >= 30) return "低め";
  return "サイド";
}

export function DesignControls({
  input,
  onChange,
  zone,
  onZone,
}: {
  input: ReleaseInput;
  onChange: (next: ReleaseInput) => void;
  zone: { bottomFt: number; topFt: number };
  onZone: (zone: { bottomFt: number; topFt: number }) => void;
}) {
  const set = (partial: Partial<ReleaseInput>) => onChange({ ...input, ...partial });
  const applyGrip = (grip: GripId) => {
    onChange(
      presetInput(grip, {
        handedness: input.handedness,
        hand: input.hand,
        pitcherHeightCm: input.pitcherHeightCm,
        extensionFt: input.extensionFt,
        aimXIn: input.aimXIn,
        aimZFt: input.aimZFt,
        tempC: input.tempC,
        elevationM: input.elevationM,
        sswScale: input.sswScale,
      }),
    );
  };

  return (
    <div className="grid gap-5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-[13px]">
          <span className={input.handedness === "R" ? "text-foreground" : "text-muted-foreground"}>右</span>
          <Switch
            checked={input.handedness === "L"}
            onCheckedChange={(checked) => set({ handedness: checked ? "L" : "R" })}
            aria-label="利き腕"
          />
          <span className={input.handedness === "L" ? "text-foreground" : "text-muted-foreground"}>左</span>
        </div>
        <Button type="button" variant="outline" size="sm" onClick={() => applyGrip(input.grip)}>
          この球種に戻す
        </Button>
      </div>

      <div className="grid grid-cols-4 gap-1.5">
        {GRIP_LIST.map((grip) => (
          <button
            key={grip.id}
            type="button"
            onClick={() => applyGrip(grip.id)}
            className={`rounded-md border px-1.5 py-1.5 text-left text-[11px] leading-tight ${
              input.grip === grip.id ? "border-[var(--gold)] bg-accent" : "border-border bg-card hover:bg-accent"
            }`}
          >
            <span className="mb-1 block h-1 rounded-full" style={{ background: grip.color }} />
            {grip.name}
          </button>
        ))}
      </div>

      <GripDiagram grip={input.grip} fingers={input.fingers} />

      <section className="grid gap-3">
        <h3 className="text-[12px] tracking-wide text-muted-foreground">指の力（その指の上限に対する％）</h3>
        {FINGERS.map((finger) => (
          <ParamSlider
            key={finger.id}
            label={finger.label}
            value={input.fingers[finger.id]}
            min={0}
            max={100}
            step={1}
            display={`${Math.round(input.fingers[finger.id])}%`}
            onChange={(value) => set({ fingers: { ...input.fingers, [finger.id]: value } })}
          />
        ))}
        <div>
          <div className="mb-1.5 text-[13px]">最後に離れる指</div>
          <div className="flex flex-wrap gap-1">
            {CONTACTS.map((contact) => (
              <button
                key={contact.id}
                type="button"
                onClick={() => set({ lastContact: contact.id })}
                className={`rounded-md border px-2 py-1 text-[12px] ${
                  input.lastContact === contact.id ? "border-[var(--gold)] bg-accent" : "border-border"
                }`}
              >
                {contact.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="grid gap-3">
        <h3 className="text-[12px] tracking-wide text-muted-foreground">腕の振りとリリース</h3>
        <ParamSlider
          label="アームスロット"
          hint="90°が真上、0°がサイド。スピン軸は前腕とほぼ直交します。"
          value={input.armSlotDeg}
          min={5}
          max={90}
          step={1}
          display={`${input.armSlotDeg.toFixed(0)}° ${slotWord(input.armSlotDeg)}`}
          onChange={(armSlotDeg) => set({ armSlotDeg })}
        />
        <ParamSlider
          label="腕の速度（手先）"
          hint="ボールの球速ではありません。手首と指が、この速さに上乗せします。"
          value={input.armSpeedMph}
          min={52}
          max={86}
          step={0.5}
          display={`${input.armSpeedMph.toFixed(1)} mph`}
          onChange={(armSpeedMph) => set({ armSpeedMph })}
        />
        <ParamSlider
          label="回内 ＋ / 回外 −"
          hint="プラスでアームサイドに倒れ、ジャイロが増えます。左投げも同じ向きです。"
          value={input.pronationDeg}
          min={-55}
          max={55}
          step={1}
          display={`${input.pronationDeg > 0 ? "+" : ""}${input.pronationDeg.toFixed(0)}° ${input.pronationDeg >= 0 ? "回内" : "回外"}`}
          onChange={(pronationDeg) => set({ pronationDeg })}
        />
        <ParamSlider
          label="手首（伸展 ＋ / 屈曲 −）"
          hint="速球は伸展、カーブは屈曲。球速とスピン軸の両方に効きます。"
          value={input.wristExtensionDeg}
          min={-40}
          max={35}
          step={1}
          display={`${input.wristExtensionDeg > 0 ? "+" : ""}${input.wristExtensionDeg.toFixed(0)}° ${input.wristExtensionDeg >= 0 ? "伸展" : "屈曲"}`}
          onChange={(wristExtensionDeg) => set({ wristExtensionDeg })}
        />
        <ParamSlider
          label="リリースのタイミング"
          hint="マイナスは早い、プラスは遅い。遅れるとアームサイドに倒れ、効率が落ちます。"
          value={input.releaseTimingMs}
          min={-28}
          max={28}
          step={1}
          display={`${input.releaseTimingMs > 0 ? "+" : ""}${input.releaseTimingMs.toFixed(0)} ms ${
            input.releaseTimingMs === 0 ? "ちょうど" : input.releaseTimingMs > 0 ? "遅い" : "早い"
          }`}
          onChange={(releaseTimingMs) => set({ releaseTimingMs })}
        />
      </section>

      <section className="grid gap-3">
        <h3 className="text-[12px] tracking-wide text-muted-foreground">狙い・体・空気</h3>
        <ParamSlider
          label="狙いの横（捕手から見て右が＋）"
          value={input.aimXIn}
          min={-16}
          max={16}
          step={0.5}
          display={`${input.aimXIn > 0 ? "+" : ""}${input.aimXIn.toFixed(1)} in`}
          onChange={(aimXIn) => set({ aimXIn })}
        />
        <ParamSlider
          label="狙いの高さ"
          value={input.aimZFt}
          min={1}
          max={4.2}
          step={0.05}
          display={`${input.aimZFt.toFixed(2)} ft`}
          onChange={(aimZFt) => set({ aimZFt })}
        />
        <ParamSlider
          label="エクステンション"
          value={input.extensionFt}
          min={5}
          max={7.6}
          step={0.1}
          display={`${input.extensionFt.toFixed(1)} ft`}
          onChange={(extensionFt) => set({ extensionFt })}
        />
        <ParamSlider
          label="身長"
          value={input.pitcherHeightCm}
          min={160}
          max={210}
          step={1}
          display={`${input.pitcherHeightCm.toFixed(0)} cm`}
          onChange={(pitcherHeightCm) => set({ pitcherHeightCm })}
        />
        <ParamSlider
          label="気温"
          value={input.tempC}
          min={5}
          max={38}
          step={1}
          display={`${input.tempC.toFixed(0)} °C`}
          onChange={(tempC) => set({ tempC })}
        />
        <ParamSlider
          label="標高"
          value={input.elevationM}
          min={0}
          max={2200}
          step={20}
          display={`${input.elevationM.toFixed(0)} m`}
          onChange={(elevationM) => set({ elevationM })}
        />
        <ParamSlider
          label="縫い目ウェイクの強さ"
          hint="シームシフトの近似項です。0で切り、空力モデルのマグヌス力だけになります。"
          value={input.sswScale}
          min={0}
          max={1.5}
          step={0.05}
          display={`${Math.round(input.sswScale * 100)}%`}
          onChange={(sswScale) => set({ sswScale })}
        />
        <ParamSlider
          label="ゾーン上端"
          value={zone.topFt}
          min={2.6}
          max={4.2}
          step={0.05}
          display={`${zone.topFt.toFixed(2)} ft`}
          onChange={(topFt) => onZone({ ...zone, topFt })}
        />
        <ParamSlider
          label="ゾーン下端"
          value={zone.bottomFt}
          min={0.8}
          max={2.4}
          step={0.05}
          display={`${zone.bottomFt.toFixed(2)} ft`}
          onChange={(bottomFt) => onZone({ ...zone, bottomFt })}
        />
      </section>
    </div>
  );
}
