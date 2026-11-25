"""Five-finger MuJoCo release, and the full-body delivery driven by joint-center motion.

Examples:
  PYTHONPATH=python python3 -m pitchlab
  PYTHONPATH=python python3 -m pitchlab --video python/out/mujoco-release.mp4
  PYTHONPATH=python python3 -m pitchlab --sweep python/out/surrogate.json
  PYTHONPATH=python python3 -m pitchlab --body
  PYTHONPATH=python python3 -m pitchlab --body --motion csv:my_pitch.csv,1.80,80
  PYTHONPATH=python python3 -m pitchlab --population 8
  PYTHONPATH=python python3 -m pitchlab --body-video python/out/mujoco_fullbody.mp4
  PITCHLAB_OBP_ELIGIBLE=1 PYTHONPATH=python python3 -m pitchlab --body-validate 29
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pitchlab.report import compare, dump_surrogate, errors_as_dicts, format_table, sweep_four_seam
from pitchlab.simulate import simulate


def _motion(spec: str):
    """literature | literature-mean | csv:PATH[,height_m,mass_kg] | obp:SESSION_PITCH (opt-in, see obp.ELIGIBLE_ENV)."""
    from pitchlab.fullbody import default_motion

    if spec == "literature":
        return default_motion()
    if spec == "literature-mean":
        from pitchlab.literature import literature_motion

        return literature_motion(mean=True)
    kind, _, rest = spec.partition(":")
    if kind == "csv":
        from pitchlab.motion import load_csv

        path, *size = rest.split(",")
        height, mass = (float(size[0]), float(size[1])) if len(size) == 2 else (1.85, 88.0)
        return load_csv(path, height_m=height, mass_kg=mass)
    if kind == "obp":
        from pitchlab.obp import load_pitch, to_motion

        return to_motion(load_pitch(rest))
    raise SystemExit(f"unknown --motion {spec!r}")


def _body(args: argparse.Namespace) -> None:
    from pitchlab import bodyreport
    from pitchlab.fullbody import format_result, render_delivery, simulate_delivery, validate

    if args.body_validate:
        from pitchlab.obp import load_pitch, to_motion

        motions = [to_motion(load_pitch(sp)) for sp in bodyreport.pick_pitches(args.body_validate)]
        results = validate(motions)
        rows = bodyreport.validation_rows(results)
        print(bodyreport.format_validation(rows))
        if args.body_json:
            path = Path(args.body_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {path}")
        return
    if args.population:
        from pitchlab.pitcher import consistent_draws

        results = validate(consistent_draws(args.population, seed=args.seed, log=lambda s: print(s, flush=True)))
        for r in results:
            print(format_result(r).splitlines()[1])
        print()
        print(f"population of {len(results)} self-consistent pitchers (mean ± SD):")
        print(bodyreport.format_criteria(bodyreport.criteria_rows(results)))
        return
    motion = _motion(args.motion)
    if args.body_video:
        result, model, frames, ref = simulate_delivery(
            "four-seam", motion, store_frames=True, frame_dt=0.002, ball=args.ball
        )
    else:
        result = simulate_delivery("four-seam", motion, ball=args.ball)
    print(format_result(result))
    print()
    print(bodyreport.format_criteria(bodyreport.criteria_rows([result])))
    print()
    print(bodyreport.format_independent(bodyreport.independent_rows(result)))
    from pitchlab.literature import chain_check

    chain = chain_check(motion)
    print()
    print(
        "kinetic chain, kinematic residual (not the contact force): "
        f"stride {chain['stride_frac'] * 100:.1f}% height, "
        f"COM speed at foot contact {chain['com_speed_fc_m_s']:.2f} m/s, "
        f"push-off AP {chain['push_off_ap_bw']:.2f} BW (MacWilliams 0.35), "
        f"lead-hip brake {chain['lead_hip_brake_g']:.2f} g, "
        f"whole-body AP after contact {chain['landing_ap_bw']:.2f} BW"
    )
    print()
    print(bodyreport.format_literature(bodyreport.literature_rows(result, motion)))
    print()
    print(bodyreport.format_statcast(bodyreport.statcast_row(result)))
    if args.body_video:
        path = Path(args.body_video)
        path.parent.mkdir(parents=True, exist_ok=True)
        render_delivery(model, frames, str(path), ref, result, frame_dt=0.002)
        print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="5 本指の MuJoCo リリースと、関節中心の軌道で駆動する全身投球")
    parser.add_argument("--video", type=str, default="", help="フォーシームのスロー映像")
    parser.add_argument("--sweep", type=str, default="", help="感度スイープの JSON")
    parser.add_argument("--json", type=str, default="", help="4 球種の比較 JSON")
    parser.add_argument("--body", action="store_true", help="全身モデルで 1 球を再現する")
    parser.add_argument("--motion", type=str, default="literature",
                        help="literature（既定、文献 SD 内で整合させた一人の投手）| literature-mean（全項目を文献平均）| csv:PATH[,身長m,体重kg] | obp:SESSION_PITCH（要 PITCHLAB_OBP_ELIGIBLE=1）")
    parser.add_argument("--population", type=int, default=0, help="文献の平均と SD から N 人を抽出して基準を判定")
    parser.add_argument("--seed", type=int, default=0, help="--population の乱数シード")
    parser.add_argument("--body-video", type=str, default="", help="全身の横・正面・後方のスロー映像")
    parser.add_argument("--body-validate", type=int, default=0, help="OBP の右投手 N 人で検証（任意、要 PITCHLAB_OBP_ELIGIBLE=1）")
    parser.add_argument("--body-json", type=str, default="", help="検証結果の JSON")
    parser.add_argument("--ball", type=str, default="npb", choices=("npb", "npb-rosin", "mlb"),
                        help="npb（既定、統一球・無塗布）| npb-rosin（統一球に MLB のロジン上昇幅を掛けた代用）| mlb（ローリングス想定、ロジン）")
    args = parser.parse_args()

    if args.body or args.body_video or args.body_validate or args.population:
        _body(args)
        return

    rows = compare()
    print(format_table(rows))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(errors_as_dicts(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    if args.sweep:
        surrogate = sweep_four_seam()
        path = Path(args.sweep)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump_surrogate(str(path), surrogate)
        print(f"wrote {path}")
    if args.video:
        result, model, frames = simulate("four-seam", store_frames=True, frame_dt=0.0005)
        from pitchlab.simulate import render_video

        path = Path(args.video)
        path.parent.mkdir(parents=True, exist_ok=True)
        render_video(model, frames, str(path), frame_dt=0.0005)
        print(
            f"wrote {path}  {result.speed_mph:.1f} mph  {result.spin_rpm:.0f} rpm  "
            f"eff {result.spin_efficiency:.2f}"
        )


if __name__ == "__main__":
    main()
