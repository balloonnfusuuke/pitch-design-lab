"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { detectHand } from "@/lib/hand/detect";
import { FINGER_IDS, leverLengthCm, withLeverLength, type FingerId, type HandProfile, type Point3 } from "@/lib/hand/profile";

const NAMES: Record<FingerId, string> = {
  index: "人差し指",
  middle: "中指",
  ring: "薬指",
  pinky: "小指",
  thumb: "親指",
};

const LINKS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

export function HandPanel({ hand, onChange }: { hand: HandProfile; onChange: (hand: HandProfile) => void }) {
  const [referenceCm, setReferenceCm] = useState(hand.handLengthCm);
  const [preview, setPreview] = useState<string | null>(null);
  const [landmarks, setLandmarks] = useState<Point3[] | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !preview) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const image = new Image();
    image.onload = () => {
      const max = 640;
      const scale = Math.min(1, max / image.naturalWidth);
      canvas.width = Math.round(image.naturalWidth * scale);
      canvas.height = Math.round(image.naturalHeight * scale);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      if (!landmarks) return;
      ctx.strokeStyle = "#e0b15a";
      ctx.lineWidth = 2;
      for (const [a, b] of LINKS) {
        ctx.beginPath();
        ctx.moveTo(landmarks[a].x * canvas.width, landmarks[a].y * canvas.height);
        ctx.lineTo(landmarks[b].x * canvas.width, landmarks[b].y * canvas.height);
        ctx.stroke();
      }
      ctx.fillStyle = "#d94a32";
      for (const point of landmarks) {
        ctx.beginPath();
        ctx.arc(point.x * canvas.width, point.y * canvas.height, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    };
    image.src = preview;
  }, [preview, landmarks]);

  const measure = async (file: File) => {
    setBusy(true);
    setError(null);
    setWarning(null);
    try {
      const url = URL.createObjectURL(file);
      setPreview(url);
      const image = await loadImage(url);
      const found = await detectHand(image, referenceCm);
      onChange(found.hand);
      setLandmarks(found.landmarks);
      setWarning(found.warning);
    } catch (caught) {
      setLandmarks(null);
      setError(caught instanceof Error ? caught.message : "計測に失敗しました。");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-4">
      <p className="text-[12px] leading-relaxed text-muted-foreground">
        手のひらをカメラに向けた写真と、手首から中指の先までの長さ（cm）で指の節をスケールします。写真が無いときは、下の長さを直接入れてください。短い指は、同じ力の％でもトルクが小さくなります。
      </p>
      <div className="grid gap-1.5">
        <Label htmlFor="hand-length">基準の長さ（手首 → 中指の先）</Label>
        <Input
          id="hand-length"
          type="number"
          min={12}
          max={26}
          step={0.1}
          value={referenceCm}
          onChange={(event) => setReferenceCm(Number(event.target.value))}
        />
      </div>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void measure(file);
        }}
      />
      <Button type="button" variant="outline" disabled={busy} onClick={() => fileRef.current?.click()}>
        {busy ? "計測しています…" : "写真から測る"}
      </Button>
      {preview ? <canvas ref={canvasRef} className="max-h-64 w-full rounded-md border border-border object-contain" /> : null}
      {warning ? <p className="text-[12px] text-[var(--gold)]">{warning}</p> : null}
      {error ? <p className="text-[12px] text-[#d94a32]">{error}</p> : null}
      <div className="grid gap-2">
        {FINGER_IDS.map((finger) => (
          <label key={finger} className="grid grid-cols-[5.5rem_minmax(0,1fr)_3.2rem] items-center gap-2 text-[13px]">
            <span>{NAMES[finger]}</span>
            <input
              type="range"
              min={4}
              max={12}
              step={0.1}
              value={leverLengthCm(hand, finger)}
              onChange={(event) => onChange(withLeverLength(hand, finger, Number(event.target.value)))}
            />
            <span className="lab-num text-right text-[12px] text-[var(--gold)]">{leverLengthCm(hand, finger).toFixed(1)}</span>
          </label>
        ))}
      </div>
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        写真計測は MediaPipe Hands です。WASM は CDN から読みます。届かないときは上のスライダーだけで手を入れられます。長さは正面写真の近似で、骨の計測ではありません。
      </p>
    </div>
  );
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("画像を開けませんでした。"));
    image.src = url;
  });
}
