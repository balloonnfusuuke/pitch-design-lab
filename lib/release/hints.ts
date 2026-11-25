import { leverLengthCm } from "../hand/profile";
import type { PitchMetrics } from "../physics/trajectory";
import { GRIPS } from "./grips";
import type { ReleaseInput } from "./types";

export type Hint = {
  id: string;
  severity: "high" | "mid" | "info";
  text: string;
};

function armSide(metrics: PitchMetrics, hand: ReleaseInput["handedness"]): number {
  return hand === "R" ? -metrics.hbIn : metrics.hbIn;
}

function gloveSide(metrics: PitchMetrics, hand: ReleaseInput["handedness"]): number {
  return -armSide(metrics, hand);
}

export function trainingHints(input: ReleaseInput, metrics: PitchMetrics): Hint[] {
  const hints: Hint[] = [];
  const push = (hint: Hint) => hints.push(hint);
  const grip = input.grip;

  if (input.releaseTimingMs > 18) {
    push({
      id: "late",
      severity: "mid",
      text: "リリースが遅れています。スピン軸がアームサイドに倒れ、回転効率と球速が落ちます。もう半拍早く離す感覚です。",
    });
  } else if (input.releaseTimingMs < -18) {
    push({
      id: "early",
      severity: "mid",
      text: "リリースが早すぎます。指がボールに仕事をする前に離れ、回転が不安定になります。",
    });
  }

  if (input.fingers.thumb >= 82 && grip !== "splitter") {
    push({
      id: "thumb",
      severity: "mid",
      text: "親指が強すぎます。親指は支える役で、押しすぎると指先の接線力が殺されます。",
    });
  }

  const middleCm = leverLengthCm(input.hand, "middle");
  if (middleCm < 8 && (grip === "four-seam" || grip === "curve" || grip === "sweeper")) {
    push({
      id: "lever",
      severity: "info",
      text: `中指は ${middleCm.toFixed(1)} cm で、基準（9.2 cm）より短いです。同じ力の％でもトルクが小さく、回転数は伸びにくい手です。`,
    });
  }

  if (grip === "four-seam") {
    if (metrics.ivbIn < 14) {
      if (input.fingers.index < 65 || input.fingers.middle < 65) {
        push({
          id: "ff-fingers",
          severity: "high",
          text: "縦の伸びが足りません。人差し指と中指の力が弱く、バックスピンが作れていません。",
        });
      } else if (metrics.spinEfficiency < 0.82) {
        push({
          id: "ff-gyro",
          severity: "high",
          text: "回転数はあっても効率が低いです。回内が早く、ジャイロ成分に吸われています。指先は上向きの回転に使ってください。",
        });
      } else if (
        (metrics.tiltDeg > 40 && input.handedness === "R") ||
        (metrics.tiltDeg < -40 && input.handedness === "L")
      ) {
        push({
          id: "ff-tilt",
          severity: "high",
          text: "スピン軸がアームサイドに倒れすぎています。スロットを上げるか、回内を減らすと縦に戻ります。",
        });
      } else {
        push({
          id: "ff-spin",
          severity: "high",
          text: "縦の伸びがフォーシームとしては小さいです。回転数そのものを、指を最後まで残して上げてください。",
        });
      }
    }
    if (metrics.releaseSpeedMph < 90 && input.armSpeedMph < 70) {
      push({
        id: "ff-arm",
        severity: "high",
        text: "球速が不足しています。指より先に、腕（手先）の速度が足りていません。",
      });
    }
    if (armSide(metrics, input.handedness) > 13) {
      push({
        id: "ff-run",
        severity: "mid",
        text: "シュート成分が大きめです。フォーシームとして真っすぐ伸ばすなら、回内を一段抑えます。",
      });
    }
  }

  if (grip === "two-seam") {
    if (metrics.ivbIn > 14) {
      push({
        id: "si-ride",
        severity: "mid",
        text: "まだフォーシームに近い伸びがあります。スロットを下げ、回内を増やし、縫い目のウェイクを残すと沈みます。",
      });
    }
    if (armSide(metrics, input.handedness) < 8) {
      push({
        id: "si-run",
        severity: "high",
        text: "アームサイドの変化が足りません。薬指の圧力と回内、低めのスロットがシュート成分を作ります。",
      });
    }
  }

  if (grip === "cutter") {
    if (gloveSide(metrics, input.handedness) < 1.5) {
      push({
        id: "ct-cut",
        severity: "high",
        text: "カットがほぼ出ていません。人差し指をボールの外側に残し、回外（マイナス）を少し足します。",
      });
    }
  }

  if (grip === "slider") {
    if (metrics.spinEfficiency > 0.58) {
      push({
        id: "sl-gyro",
        severity: "mid",
        text: "効率が高すぎて、ジャイロスライダーというより横回転の球です。軸をもう少し進行方向に向けると、変化は小さく鋭くなります。",
      });
    }
    if (gloveSide(metrics, input.handedness) < 3) {
      push({
        id: "sl-break",
        severity: "high",
        text: "グラブサイドへの変化が足りません。スライダーの握りで人差し指の圧力と、わずかな回外を確認してください。",
      });
    }
  }

  if (grip === "sweeper") {
    if (gloveSide(metrics, input.handedness) < 10) {
      push({
        id: "sw-sweep",
        severity: "high",
        text: "スイーパーとしての横変化が足りません。スロットを下げ、中指で横に剥がし、回外を残します。効率を落としすぎるとジャイロに吸われます。",
      });
    } else if (metrics.spinEfficiency < 0.45) {
      push({
        id: "sw-eff",
        severity: "mid",
        text: "回転効率が低く、横変化がジャイロに食われています。スイーパーはスライダーより軸を横に寝かせ、効率を残します。",
      });
    }
  }

  if (grip === "curve") {
    if (metrics.ivbIn > -4) {
      push({
        id: "cb-drop",
        severity: "high",
        text: "縦に落ちていません。手首を屈曲（角度をマイナス）させ、中指でボールの上側を最後まで押します。",
      });
    }
    if (gloveSide(metrics, input.handedness) < 2 && metrics.ivbIn <= -4) {
      push({
        id: "cb-glove",
        severity: "info",
        text: "落ちてはいますが、横の曲がりは小さいです。12-6 に近い軌道です。横も欲しいなら回外を少し足します。",
      });
    }
  }

  if (grip === "changeup") {
    if (metrics.releaseSpeedMph > 89) {
      push({
        id: "ch-velo",
        severity: "high",
        text: "チェンジアップとしては速すぎます。腕の速度は速球のまま、指を深くして手首の加速だけを殺してください。",
      });
    }
    if (metrics.spinRpm > 2100) {
      push({
        id: "ch-spin",
        severity: "mid",
        text: "回転が落ちていません。薬指と小指をボールに食い込ませ、人差し指の力を抜くと回転数は下がります。",
      });
    }
    if (armSide(metrics, input.handedness) < 6) {
      push({
        id: "ch-run",
        severity: "mid",
        text: "アームサイドへの動きが小さいです。回内を速球より強くすると、チェンジアップらしいシュートが出ます。",
      });
    }
  }

  if (grip === "splitter") {
    if (metrics.spinRpm > 1700) {
      push({
        id: "sp-spin",
        severity: "high",
        text: "まだ回転が多いです。人差し指と中指を広げ、接線方向の力を抜かないとスプリットは沈みません。",
      });
    } else if (metrics.ivbIn > 8) {
      push({
        id: "sp-ride",
        severity: "mid",
        text: "落ち切っていません。低めの回転効率と、縫い目ウェイク（下方向）がスプリットの沈みを作ります。",
      });
    }
  }

  if (hints.length === 0) {
    push({
      id: "ok",
      severity: "info",
      text: `${GRIPS[grip].name}として、球速・回転・変化のバランスは狙いに近いです。比較に重ねて、判断点で他の球と重なるかを見てください。`,
    });
  }

  const rank = { high: 0, mid: 1, info: 2 };
  return hints.sort((a, b) => rank[a.severity] - rank[b.severity]).slice(0, 4);
}
