"""The five-finger release stays finite and uses the cited parameters."""

from __future__ import annotations

import math
import unittest

import mujoco
import numpy as np

from pitchlab.balls import BALLS, DEFAULT_BALL_ID, ROSIN_LEATHER_GAIN, ROSIN_SEAM_GAIN, contact_mu
from pitchlab.constants import (
    CONTACT_MS,
    ELBOW_EXTENSION_DEG_S,
    ELBOW_FLEXION_AT_RELEASE_DEG,
    FRICTION,
    MCP_TORQUE_NM,
    PAD_RADIUS_M,
    PULP_NODE_RADIUS_M,
    PULP_PROTRUSION_M,
    PULP_TRAVEL_M,
    SEAM_COUNT,
    SEAM_FRICTION,
    SEAM_PROTRUSION_M,
    SHOULDER_IR_DEG_S,
    WRIST_FLEX_DEG_S,
)
from pitchlab.report import REFERENCES, plate_break_inches
from pitchlab.seams import seam_samples
from pitchlab.grips import GRIPS
from pitchlab.simulate import BELLY_LOCAL_M, PULP_CENTERS, PULP_FINGERS, PULP_TIP_CENTERS, build_model, simulate, wrist_state


class LiteratureParameters(unittest.TestCase):
    def test_cited_constants(self) -> None:
        self.assertEqual(SHOULDER_IR_DEG_S, 7511.0)
        self.assertEqual(ELBOW_EXTENSION_DEG_S, 2438.0)
        self.assertEqual(ELBOW_FLEXION_AT_RELEASE_DEG, 22.0)
        self.assertEqual(WRIST_FLEX_DEG_S, 2954.0)
        self.assertEqual(CONTACT_MS, 8.0)
        self.assertEqual(DEFAULT_BALL_ID, "npb")
        self.assertAlmostEqual(FRICTION, 0.9585)
        self.assertAlmostEqual(SEAM_FRICTION, 1.0165)
        self.assertAlmostEqual(SEAM_PROTRUSION_M, 0.00090)
        self.assertEqual(MCP_TORQUE_NM, 6.4)

    def test_mlb_preset_keeps_the_rosin_ball(self) -> None:
        mlb = BALLS["mlb"]
        self.assertAlmostEqual(mlb.mu_leather, 0.965)
        self.assertAlmostEqual(mlb.mu_seam, 1.107)
        self.assertAlmostEqual(mlb.seam_height_m, 0.00080)
        self.assertIsNone(mlb.cor_target)
        npb = BALLS["npb"]
        self.assertAlmostEqual(npb.seam_height_m, 0.00090)
        self.assertAlmostEqual(npb.cor_target, 0.4134)
        self.assertAlmostEqual(npb.mass_kg, mlb.mass_kg)
        # An explicit leather μ scales the seam; the bat COR is not a friction.
        spec, mu, seam = contact_mu("mlb", None)
        self.assertEqual(spec.id, "mlb")
        self.assertAlmostEqual(mu, 0.965)
        self.assertAlmostEqual(seam, 1.107)
        _, mu_s, seam_s = contact_mu("npb", None)
        self.assertAlmostEqual(mu_s, 0.9585)
        self.assertAlmostEqual(seam_s, 1.0165)
        rosin = BALLS["npb-rosin"]
        self.assertAlmostEqual(rosin.mu_leather, round(0.9585 * ROSIN_LEATHER_GAIN, 4))
        self.assertAlmostEqual(rosin.mu_seam, round(1.0165 * ROSIN_SEAM_GAIN, 4))
        self.assertAlmostEqual(rosin.seam_height_m, 0.00090)
        self.assertIn("substitute", rosin.mu_condition)

    def test_savant_targets_are_2024_rhp(self) -> None:
        ff = REFERENCES["four-seam"]
        self.assertAlmostEqual(ff["mph"], 94.72)
        self.assertAlmostEqual(ff["rpm"], 2312.0)
        self.assertAlmostEqual(ff["ivb"], 15.75)
        self.assertLess(ff["hb"], 0)
        self.assertGreater(REFERENCES["curve"]["hb"], 0)
        self.assertLess(REFERENCES["slider"]["eff"], 0.5)
        self.assertGreater(REFERENCES["curve"]["rpm"], REFERENCES["four-seam"]["rpm"])


class FourSeamGrip(unittest.TestCase):
    def test_fingertips_cross_one_horseshoe_and_the_thumb_is_on_leather(self) -> None:
        from pitchlab.fourseam import grip_report
        from pitchlab.fullbody import _body, _fingertip_grip

        grip = GRIPS["four-seam"]
        model = build_model(grip, pulp=False)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        _fingertip_grip(model, data, grip)
        hand = _body(model, "hand")
        r_hand = np.array(data.xmat[hand]).reshape(3, 3)
        adr = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")])
        center = data.qpos[adr : adr + 3]
        # The quaternion written by the grip is R_hand @ R_ball.
        ball = np.zeros(9)
        mujoco.mju_quat2Mat(ball, data.qpos[adr + 3 : adr + 7])
        rotation = r_hand.T @ ball.reshape(3, 3)
        dirs, axes = {}, {}
        for name in ("index", "middle", "thumb"):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_pad")
            dirs[name] = r_hand.T @ (data.site_xpos[sid] - center)
            axes[name] = r_hand.T @ np.array(data.site_xmat[sid]).reshape(3, 3)[:, 0]
        prox = {}
        for name in ("index", "middle"):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_pad")
            rf = np.array(data.site_xmat[sid]).reshape(3, 3)
            prox[name] = r_hand.T @ (data.site_xpos[sid] + rf @ BELLY_LOCAL_M - center)
        report = grip_report(
            rotation, dirs["index"], dirs["middle"], axes["index"], axes["middle"], dirs["thumb"],
            prox["index"], prox["middle"],
        )
        self.assertLess(report["index_seam_deg"], 2.0)
        self.assertLess(report["middle_seam_deg"], 2.0)
        # Seam tangent nearly perpendicular to the finger (1 would be parallel).
        self.assertLess(report["index_cross"], 0.45)
        self.assertLess(report["middle_cross"], 0.45)
        self.assertGreater(report["thumb_seam_deg"], 20.0)
        self.assertGreater(report["other_leg_proximal"], 0.0)
        # The other leg of this horseshoe does not land on both distal pads.
        # The proximal point is reported, not required to sit on a seam.
        self.assertIn("index_prox_seam_deg", report)
        self.assertIn("middle_prox_seam_deg", report)


class PulpGeometry(unittest.TestCase):
    def test_pulp_sits_on_the_old_envelope_and_can_drop_past_the_seam(self) -> None:
        # Travel clears the measured seam height, so a node beside the bump can
        # stay proud after its neighbors bottom. The free surface is the 9 mm
        # sphere the grip solver still aims at.
        self.assertGreater(PULP_TRAVEL_M, SEAM_PROTRUSION_M)
        self.assertGreaterEqual(len(PULP_TIP_CENTERS), 8)
        self.assertTrue(any(float(c[0]) < -0.008 for c in PULP_CENTERS))
        for center in PULP_TIP_CENTERS:
            self.assertAlmostEqual(
                float(np.linalg.norm(center)) + PULP_NODE_RADIUS_M,
                PAD_RADIUS_M + PULP_PROTRUSION_M,
                places=5,
            )
        model = build_model(GRIPS["four-seam"])
        slides = [k for k in range(model.njnt) if int(model.jnt_type[k]) == int(mujoco.mjtJoint.mjJNT_SLIDE)]
        self.assertGreaterEqual(len(slides), len(PULP_FINGERS) * len(PULP_CENTERS))
        self.assertLess(len(slides), 5 * len(PULP_CENTERS))
        self.assertEqual(sum(1 for i in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("seam_")), SEAM_COUNT)
        seam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "seam_0")
        node = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "index_pad_00")
        backing = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "index_pad")
        # An in-between bump is paired with the node and not with the backing.
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "seam_1")
        node_mid = backing_mid = node_0 = backing_0 = False
        for i in range(model.npair):
            geoms = {int(model.pair_geom1[i]), int(model.pair_geom2[i])}
            node_mid = node_mid or geoms == {mid, node}
            backing_mid = backing_mid or geoms == {mid, backing}
            node_0 = node_0 or geoms == {seam, node}
            backing_0 = backing_0 or geoms == {seam, backing}
            if geoms == {seam, node} and float(model.pair_solimp[i, 2]) < 0.0005:
                self.assertAlmostEqual(float(model.pair_friction[i, 0]), SEAM_FRICTION)
        self.assertTrue(node_mid and node_0 and backing_0)
        self.assertFalse(backing_mid)
        # The bat COR is a rebound target, not the finger-contact time constant.
        leather = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "leather")
        self.assertAlmostEqual(float(model.geom_solref[leather][0]), 0.004)
        self.assertAlmostEqual(float(model.geom_friction[leather][0]), FRICTION)
        self.assertAlmostEqual(float(model.geom_size[leather][0]), BALLS["npb"].radius_m, places=6)
        mlb = build_model(GRIPS["four-seam"], ball="mlb")
        mlb_leather = mujoco.mj_name2id(mlb, mujoco.mjtObj.mjOBJ_GEOM, "leather")
        self.assertAlmostEqual(float(mlb.geom_friction[mlb_leather][0]), 0.965)
        mlb_seam = mujoco.mj_name2id(mlb, mujoco.mjtObj.mjOBJ_GEOM, "seam_0")
        self.assertAlmostEqual(float(mlb.geom_friction[mlb_seam][0]), 1.107)


class ReleasePhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = {name: simulate(name) for name in ("four-seam", "curve", "slider", "changeup")}

    def test_wrist_and_ball_move_toward_home(self) -> None:
        vel, speed = wrist_state()
        self.assertGreater(speed, 10.0)
        self.assertGreater(vel[1], abs(float(vel[0])))
        self.assertGreater(vel[1], abs(float(vel[2])))
        for name, result in self.results.items():
            speed_b = float(np.linalg.norm(result.velocity))
            self.assertGreater(result.velocity[1] / speed_b, 0.85, name)
            self.assertGreater(result.velocity[1], 12.0, name)
            self.assertLess(abs(result.velocity[2]), 8.0, name)
            self.assertTrue(result.separated, name)
            self.assertFalse(np.isnan(result.omega).any(), name)
            # PIP/DIP cannot fold backward. MCP may extend a few degrees.
            self.assertGreater(result.min_finger_rad, -0.16, name)
            self.assertGreater(result.forearm_up, -0.2, name)
            self.assertGreater(result.ahead_of_head_m, 0.05, name)

    def test_four_seam_palm_faces_home(self) -> None:
        result = self.results["four-seam"]
        self.assertGreater(result.palm_forward, 0.9)
        self.assertGreaterEqual(result.peak_contacts, 3)
        self.assertGreater(result.spin_rpm, 400)
        # Thumb and the early fingers are off before index/middle finish.
        self.assertLess(result.finger_last_ms["thumb"], result.finger_last_ms["index"])

    def test_slider_is_more_gyro_than_the_fastball(self) -> None:
        fast = self.results["four-seam"]
        self.assertLess(self.results["slider"].spin_efficiency, fast.spin_efficiency)
        self.assertLess(self.results["changeup"].spin_rpm, fast.spin_rpm)

    def test_sweep_override_does_not_leak(self) -> None:
        simulate("four-seam", friction=0.65, contact_ms=7.0, ir_deg_s=6000.0)
        again = simulate("four-seam")
        self.assertAlmostEqual(again.speed_mph, self.results["four-seam"].speed_mph, delta=0.05)
        self.assertEqual(CONTACT_MS, 8.0)
        self.assertEqual(SHOULDER_IR_DEG_S, 7511.0)

    def test_seams_lie_on_the_sphere(self) -> None:
        for point in seam_samples(36):
            self.assertAlmostEqual(float(np.linalg.norm(point)), 1.0, places=6)

    def test_plate_break_is_finite(self) -> None:
        ivb, hb = plate_break_inches(self.results["four-seam"].velocity, self.results["four-seam"].omega)
        self.assertTrue(math.isfinite(ivb))
        self.assertTrue(math.isfinite(hb))
        self.assertGreater(abs(ivb), 5)


if __name__ == "__main__":
    unittest.main()
