"""Full-body model: helpers, the literature motion, CSV input, the OBP gate, and end-to-end runs."""

from __future__ import annotations

import math
import os
import tempfile
import unittest

import numpy as np

from pitchlab import obp
from pitchlab.filters import lowpass
from pitchlab.imitate import Spline, quat_conj, quat_from_mat, quat_log, quat_mul
from pitchlab.literature import TRUNK_FWD_BR, TRUNK_LAT_BR, literature_motion, single_pitcher_z
from pitchlab.mocap import landmarks_from_mediapipe
from pitchlab.motion import POINTS, load_csv, save_csv


class Filter(unittest.TestCase):
    def test_forward_backward_butterworth_is_half_power_squared_at_cutoff(self) -> None:
        fs, fc = 2000.0, 20.0
        t = np.arange(0, 2.0, 1 / fs)
        for f, expected in ((2.0, 1.0), (fc, 0.5), (200.0, 0.0)):
            y = lowpass(np.sin(2 * math.pi * f * t), fs, fc)
            gain = np.max(np.abs(y[len(t) // 4 : -len(t) // 4]))
            with self.subTest(f=f):
                self.assertAlmostEqual(gain, expected, delta=0.03)


class Quaternions(unittest.TestCase):
    def test_round_trip_and_log(self) -> None:
        a = 0.7
        R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
        q = quat_from_mat(R)
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=9)
        np.testing.assert_allclose(quat_log(q), [0, 0, a], atol=1e-9)
        ident = quat_mul(q, quat_conj(q))
        np.testing.assert_allclose(np.abs(ident), [1, 0, 0, 0], atol=1e-9)


class SplineFit(unittest.TestCase):
    def test_interpolates_and_differentiates_a_smooth_curve(self) -> None:
        x = np.linspace(0, 1, 41)
        s = Spline(x, np.sin(3 * x))
        xq = np.linspace(0.1, 0.9, 17)
        np.testing.assert_allclose(s(xq)[:, 0], np.sin(3 * xq), atol=1e-5)
        np.testing.assert_allclose(s(xq, 1)[:, 0], 3 * np.cos(3 * xq), atol=1e-3)


class MediaPipeAdapter(unittest.TestCase):
    def test_produces_every_joint_center(self) -> None:
        rng = np.random.default_rng(0)
        out = landmarks_from_mediapipe(rng.normal(size=(5, 33, 3)))
        for name in POINTS:
            with self.subTest(point=name):
                self.assertEqual(out[name].shape, (5, 3))
        with self.assertRaises(ValueError):
            landmarks_from_mediapipe(np.zeros((5, 17, 3)))


class LiteratureMotion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = literature_motion(mean=True)

    def _at(self, name: str, t: float) -> np.ndarray:
        i = int(np.argmin(np.abs(self.m.time - t)))
        return self.m.points[name][i]

    def test_thorax_has_the_published_tilt_at_release(self) -> None:
        z = self._at("thorax_prox", self.m.release) - self._at("thorax_dist", self.m.release)
        self.assertAlmostEqual(math.degrees(math.atan2(z[1], z[2])), TRUNK_FWD_BR[0], delta=1.0)
        self.assertAlmostEqual(math.degrees(math.atan2(-z[0], z[2])), TRUNK_LAT_BR[0], delta=1.0)

    def test_hip_to_thorax_line_has_the_published_tilt_too(self) -> None:
        hips = 0.5 * (self._at("lead_hip", self.m.release) + self._at("rear_hip", self.m.release))
        line = self._at("thorax_prox", self.m.release) - hips
        self.assertAlmostEqual(math.degrees(math.atan2(line[1], line[2])), TRUNK_FWD_BR[0], delta=1.0)
        self.assertAlmostEqual(math.degrees(math.atan2(-line[0], line[2])), TRUNK_LAT_BR[0], delta=1.0)

    def test_segment_lengths_are_constant(self) -> None:
        for a, b in (("shoulder_jc", "elbow_jc"), ("elbow_jc", "wrist_jc"), ("lead_hip", "lead_knee_jc")):
            d = np.linalg.norm(self.m.points[a] - self.m.points[b], axis=1)
            with self.subTest(segment=(a, b)):
                self.assertLess(float(np.ptp(d)), 1e-6)

    def test_stride_and_leg_lift_are_in_the_delivery(self) -> None:
        from pitchlab.literature import chain_check

        c = chain_check(self.m)
        # Fleisig ISBS 2010: stride 83 ± 4% of height at front-foot contact.
        self.assertGreater(c["stride_frac"], 0.79)
        self.assertLess(c["stride_frac"], 0.87)
        self.assertGreater(c["lead_ankle_z_max_m"], 0.6)
        self.assertLess(c["time_start_s"], -0.8)
        self.assertAlmostEqual(c["pelvis_open_fc_deg"], 33.0, delta=10.0)
        self.assertAlmostEqual(c["trunk_forward_fc_deg"], 4.0, delta=7.0)

    def test_hand_is_fast_near_release(self) -> None:
        v = np.linalg.norm(np.gradient(self.m.points["hand_jc"], self.m.time, axis=0), axis=1)
        window = np.abs(self.m.time - self.m.release) < 0.02
        self.assertGreater(float(np.max(v[window])), 30.0)

    def test_wrist_extension_is_inside_barrentine(self) -> None:
        from pitchlab.literature import WRIST_EXTENSION_AT_MER_DEG

        # Barrentine et al. (1998), fastball peak wrist extension 41 ± 8°.
        self.assertGreaterEqual(WRIST_EXTENSION_AT_MER_DEG, 33.0)
        self.assertLessEqual(WRIST_EXTENSION_AT_MER_DEG, 49.0)

    def test_single_pitcher_stays_within_the_published_spread(self) -> None:
        z = single_pitcher_z()
        self.assertIsNotNone(z)
        self.assertLessEqual(float(np.max(np.abs(z))), 2.55)
        self.assertLess(float(np.sum(z * z)), len(z))

    def test_population_draws_differ(self) -> None:
        a, b = (literature_motion(np.random.default_rng(s)) for s in (1, 2))
        self.assertNotAlmostEqual(a.release, b.release, places=4)


class CsvRoundTrip(unittest.TestCase):
    def test_points_and_events_survive(self) -> None:
        m = literature_motion()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pitch.csv")
            save_csv(m, path)
            back = load_csv(path, height_m=m.height_m, mass_kg=m.mass_kg)
        self.assertAlmostEqual(back.release, m.release, places=6)
        self.assertAlmostEqual(back.mer, m.mer, places=6)
        for name in POINTS:
            with self.subTest(point=name):
                np.testing.assert_allclose(back.points[name], m.points[name], atol=1e-5)


class ObpGate(unittest.TestCase):
    def test_obp_is_off_unless_the_user_opts_in(self) -> None:
        old = os.environ.pop(obp.ELIGIBLE_ENV, None)
        try:
            self.assertFalse(obp.enabled())
            with self.assertRaises(obp.OBPNotEnabled):
                obp.load_pitch(obp.DEFAULT_PITCH)
        finally:
            if old is not None:
                os.environ[obp.ELIGIBLE_ENV] = old


class LiteratureDelivery(unittest.TestCase):
    """End to end from published means only (about a minute)."""

    @classmethod
    def setUpClass(cls) -> None:
        from pitchlab.fullbody import simulate_delivery

        cls.r = simulate_delivery()

    def test_tracks_the_reference(self) -> None:
        self.assertLess(self.r.tracking_rms_deg, 1.5)
        self.assertAlmostEqual(self.r.mer_deg, self.r.reference_mer_deg, delta=3.0)

    def test_ball_leaves_the_fingertips(self) -> None:
        self.assertTrue(self.r.separated)
        self.assertGreater(self.r.velocity[1], 25.0)
        self.assertLess(self.r.release_ms, self.r.obp_release_ms + 10.0)

    def test_roll_window_is_open_loop(self) -> None:
        drive = self.r.distal_drive
        self.assertEqual(drive["mode"], "forward")
        # Pronation is a passive whip. Wrist flexion and ulnar deviation add
        # an open-loop EMG torque, still inside Delp. Holzbaur: forearm
        # 8 N·m, elbow −52 to 77 N·m.
        self.assertLessEqual(abs(drive["applied_peak_nm"]["wrist_flex"]), 18.7 + 1e-6)
        self.assertLessEqual(abs(drive["applied_peak_nm"]["wrist_dev"]), 15.3 + 1e-6)
        self.assertLessEqual(abs(drive["applied_peak_nm"]["pronation"]), 8.0 + 1e-6)
        self.assertLessEqual(drive["applied_peak_nm"]["elbow"], 77.0 + 1e-6)
        self.assertGreaterEqual(drive["applied_peak_nm"]["elbow"], -52.0 - 1e-6)
        self.assertLess(drive["above_isometric_fraction"]["wrist_dev"], 0.5)
        self.assertLess(drive["above_isometric_fraction"]["pronation"], 0.5)

    def test_feet_load_the_mound(self) -> None:
        # Stance is the set, before the knee lift. Plant is the first 80 ms
        # after the lead foot is down. Both are contact forces, not the
        # kinematic m·a residual.
        drive = self.r.distal_drive
        self.assertGreater(drive["stance_fz_set_bw"], 0.005)
        self.assertGreater(drive["lead_fz_plant_bw"], 0.05)


@unittest.skipUnless(obp.enabled() and obp._cache_file(obp.DEFAULT_PITCH).exists(),
                     f"OBP validation is opt-in ({obp.ELIGIBLE_ENV}=1) and needs the cached pitch")
class ObpPitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from pitchlab.fullbody import simulate_delivery

        cls.r = simulate_delivery(motion=obp.to_motion(obp.load_pitch()))

    def test_tracks_the_reference(self) -> None:
        self.assertLess(self.r.tracking_rms_deg, 1.0)

    def test_loads_match_obp_inverse_dynamics(self) -> None:
        for key in ("shoulder IR moment", "elbow varus moment"):
            model, ref = self.r.loads[key][0], self.r.obp_loads[key][0]
            with self.subTest(load=key):
                self.assertLess(abs(model - ref) / ref, 0.25)


if __name__ == "__main__":
    unittest.main()
