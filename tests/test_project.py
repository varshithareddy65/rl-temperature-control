"""Run:  python -m unittest discover -s tests -v   (from the project root)"""
import os, sys, tempfile, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thermal_model import (ThermalPlant, load_config, make_scenario, random_scenario, rollout)
from pid_controller import PIDController
from evaluate import compute_metrics, run_all

CFG = load_config()
try:
    import gymnasium  # noqa
    HAVE_GYM = True
except ImportError:
    HAVE_GYM = False


class BangBang:  # test-only stand-in for a second controller (never reported as a result)
    def reset(self): pass
    def act(self, obs, target, T): return 1.0 if T < target else 0.0


class TestPlant(unittest.TestCase):
    def test_steady_state_and_time_constant(self):
        p = ThermalPlant(CFG["plant"])
        T_ss = 20 + p.R * p.P_max * 0.5
        T = 20.0
        for _ in range(int(p.tau)):           # after one tau: 63.2% of the way
            T = p.step(T, 0.5, 20.0)
        self.assertAlmostEqual((T - 20) / (T_ss - 20), 1 - np.exp(-1), places=3)
        for _ in range(int(10 * p.tau)):
            T = p.step(T, 0.5, 20.0)
        self.assertAlmostEqual(T, T_ss, places=3)

    def test_cooling_and_clipping(self):
        p = ThermalPlant(CFG["plant"])
        self.assertLess(p.step(50.0, 0.0, 20.0), 50.0)
        self.assertEqual(p.step(30, 5.0, 20), p.step(30, 1.0, 20))   # u clipped to 1
        self.assertGreater(p.step(30, 0.5, 20, d=200), p.step(30, 0.5, 20, d=0))


class TestScenarios(unittest.TestCase):
    def test_random_scenarios_feasible(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            sc = random_scenario(rng, CFG)
            kmax = CFG["plant"]["R"] * CFG["plant"]["P_max"] * sc.capacity_scale
            self.assertEqual(len(sc.target), sc.n + 1)
            self.assertTrue(np.all(sc.target <= sc.ambient.min() + kmax))   # reachable at full power
            self.assertTrue(np.all(sc.target >= sc.ambient.max()))


class TestPID(unittest.TestCase):
    def test_tracks_constant_target(self):
        sc = make_scenario(CFG, 25, [(0, 50)], n=500)
        m = compute_metrics(rollout(PIDController.from_config(CFG), sc, CFG), CFG)
        self.assertLess(m["ss_error"], 0.1)
        self.assertLess(m["overshoot_pct"], 5)
        self.assertEqual(m["unsettled_segments"], 0)

    def test_antiwindup(self):
        pid = PIDController.from_config(CFG)
        for _ in range(500):
            pid.act(None, 100.0, 20.0)             # unreachable target -> saturated
        self.assertLess(pid.integral * pid.Ki, 1.5)


class TestMetrics(unittest.TestCase):
    def _res(self, T, tgt):
        n = len(T)
        return {"T": np.array(T), "target": np.array(tgt), "u": np.zeros(n), "d": np.zeros(n),
                "ambient": np.full(n, 20.0), "reward": np.zeros(n), "tripped": False}

    def test_overshoot_and_settling(self):
        T = [20.0] + list(np.linspace(20, 55, 30)) + list(np.linspace(55, 50, 20)) + [50.0] * 49
        m = compute_metrics(self._res(T, [50.0] * len(T)), CFG)
        self.assertAlmostEqual(m["overshoot_pct"], 5 / 30 * 100, places=1)
        self.assertTrue(30 < m["settling_time"] < 80)
        self.assertAlmostEqual(m["ss_error"], 0.0, places=6)

    def test_unsettled_flagged(self):
        m = compute_metrics(self._res([20.0] * 100, [50.0] * 100), CFG)
        self.assertEqual(m["unsettled_segments"], 1)


class TestPipeline(unittest.TestCase):
    def test_two_controller_run_and_plots(self):
        import visualization as viz
        cfg = load_config(); cfg["eval"]["results_dir"] = tempfile.mkdtemp()
        cfg["ppo"]["monitor_path"] = os.path.join(cfg["eval"]["results_dir"], "none.csv")
        res = run_all({"PID": PIDController.from_config(cfg), "PPO": BangBang()}, cfg)
        viz.make_all_plots(res, cfg)
        for f in ("exp1_constant_target.png", "exp2_changing_target.png", "exp3_disturbance_response.png",
                  "exp4_initial_temperatures.png", "exp5_heating_capacity.png", "comparison_metrics.png"):
            self.assertGreater(os.path.getsize(os.path.join(cfg["eval"]["results_dir"], f)), 5000, f)

    def test_training_plot_from_monitor_csv(self):
        import visualization as viz
        d = tempfile.mkdtemp(); p = os.path.join(d, "m.csv")
        with open(p, "w") as f:
            f.write('#{"t_start": 0}\nr,l,t\n' + "\n".join(f"{-100 + i * 0.1},400,{i}" for i in range(120)))
        viz.plot_training(p, os.path.join(d, "o.png"))
        self.assertTrue(os.path.exists(os.path.join(d, "o.png")))


@unittest.skipUnless(HAVE_GYM, "gymnasium not installed")
class TestEnv(unittest.TestCase):
    def setUp(self):
        from environment import TemperatureControlEnv
        self.Env = TemperatureControlEnv

    def test_reset_step_contract(self):
        env = self.Env(CFG)
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)
        n, trunc = 0, False
        while not trunc:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            self.assertTrue(np.all(np.isfinite(obs)) and np.isfinite(r))
            self.assertFalse(term)
            n += 1
        self.assertEqual(n, CFG["env"]["episode_steps"])

    def test_safety_termination(self):
        cfg = load_config(); cfg["plant"]["T_safe_max"] = 30.0
        env = self.Env(cfg, scenario=make_scenario(cfg, 29.0, [(0, 50)], n=100)); env.reset()
        term = False
        for _ in range(100):
            _, r, term, trunc, _ = env.step([1.0])
            if term: break
        self.assertTrue(term); self.assertLess(r, -cfg["reward"]["safety_penalty"] + 1)

    def test_seed_reproducible(self):
        a, b = self.Env(CFG), self.Env(CFG)
        oa, _ = a.reset(seed=5); ob_, _ = b.reset(seed=5)
        self.assertTrue(np.allclose(oa, ob_))

    def test_env_matches_rollout(self):
        sc = make_scenario(CFG, 30, [(0, 45), (100, 50)], n=300, dist=[(120, 200, -200.0)], seed=9)
        env = self.Env(CFG, scenario=sc); env.reset()
        acts = np.random.default_rng(1).uniform(-1, 1, 300)
        env_T, env_R = [], 0.0
        for a in acts:
            env_T.append(env.T); _, r, *_ = env.step([a]); env_R += r

        class Replay:
            i = 0
            def reset(self): self.i = 0
            def act(self, o, t, T):
                a = acts[self.i]; self.i += 1; return (a + 1) / 2
        res = rollout(Replay(), sc, CFG)
        self.assertTrue(np.allclose(res["T"], env_T))
        self.assertAlmostEqual(res["reward"].sum(), env_R, places=4)

    def test_sb3_check_env(self):
        try:
            from stable_baselines3.common.env_checker import check_env
        except ImportError:
            self.skipTest("stable-baselines3 not installed")
        check_env(self.Env(CFG), warn=True)


# ------------------------------------------------------------------ audit additions
class TestAuditChanges(unittest.TestCase):
    def test_default_observation_unchanged_from_v1(self):
        """dT_alpha=1 must reproduce the original observation exactly (keeps v1 models valid)."""
        from thermal_model import ObsBuilder
        ob = ObsBuilder(1.0); rng = np.random.default_rng(0)
        T_prev, e_int, prev_u = None, 0.0, 0.0
        for _ in range(50):
            Tm, tgt, amb, u = rng.uniform(20, 60), 45.0, 20.0, rng.uniform()
            e = tgt - Tm; e_int = float(np.clip(e_int + e / 50.0, -1, 1))
            dT = 0.0 if T_prev is None else Tm - T_prev; T_prev = Tm
            ref = np.array([e / 10, (Tm - 40) / 20, (tgt - 40) / 20, (amb - 20) / 10, dT / 0.5, e_int, 2 * prev_u - 1], np.float32)
            self.assertTrue(np.allclose(ob.build(Tm, tgt, amb), ref))
            ob.commit(u); prev_u = u

    def test_dT_filter_reduces_noise(self):
        from thermal_model import ObsBuilder
        rng = np.random.default_rng(1); n = rng.normal(0, 0.05, 5000)
        std = []
        for alpha in (1.0, 0.2):
            ob = ObsBuilder(1.0, alpha); std.append(np.std([ob.build(40 + x, 40, 20)[4] for x in n]))
        self.assertLess(std[1], 0.6 * std[0])

    def test_config_overrides(self):
        c = load_config("config.yaml", ["reward.w_smooth=0.3", "obs.dT_alpha=0.2"])
        self.assertEqual(c["reward"]["w_smooth"], 0.3); self.assertEqual(c["obs"]["dT_alpha"], 0.2)
        self.assertEqual(load_config()["reward"]["w_smooth"], 0.02)   # default untouched

    def test_reward_components_sum(self):
        from thermal_model import reward_components, compute_reward
        c = reward_components(42, 45, 0.6, 0.5, CFG["reward"])
        self.assertAlmostEqual(-sum(c), compute_reward(42, 45, 0.6, 0.5, CFG["reward"]))
        self.assertAlmostEqual(c[2], CFG["reward"]["w_smooth"] * 0.1)

    def test_na_semantics(self):
        """N/A (None) only where genuinely not applicable; numeric where applicable."""
        res = run_all({"PID": PIDController.from_config(CFG)}, CFG)
        for exp, cases in res.items():
            for cs in cases:
                m = cs["metrics"]["PID"]
                if exp == "exp3_disturbance":
                    self.assertIsNone(m["overshoot_pct"]); self.assertIsNone(m["settling_time"])
                    self.assertIsNotNone(m["dist_max_dev"]); self.assertIsNotNone(m["dist_recovery_time"])
                else:
                    self.assertIsNone(m["dist_max_dev"]); self.assertIsNone(m["dist_recovery_time"])
                    self.assertIsNotNone(m["overshoot_pct"]); self.assertIsNotNone(m["settling_time"])
                for k, v in m.items():
                    if isinstance(v, float): self.assertFalse(np.isnan(v), (exp, k))   # no NaN anywhere

    def test_csv_writes_na_not_nan(self):
        import evaluate, csv as _csv
        cfg = load_config(); cfg["eval"]["results_dir"] = tempfile.mkdtemp()
        res = run_all({"PID": PIDController.from_config(cfg)}, cfg)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate.print_and_save(res, cfg)
        txt = open(os.path.join(cfg["eval"]["results_dir"], "metrics.csv")).read()
        self.assertIn("N/A", txt); self.assertNotIn("nan", txt.lower())

    def test_control_var_metrics(self):
        sc = make_scenario(CFG, 25, [(0, 50)], n=400)
        res = rollout(PIDController.from_config(CFG), sc, CFG)
        m = compute_metrics(res, CFG)
        self.assertAlmostEqual(m["control_var"], m["control_tv"] / (len(res["u"]) - 1), places=9)
        self.assertGreater(m["control_var_ss"], 0)

    def test_train_backup_helper(self):
        from train import backup_if_exists
        d = tempfile.mkdtemp(); f = os.path.join(d, "m.zip"); open(f, "wb").write(b"old")
        backup_if_exists(f, os.path.join(d, "backup"))
        files = os.listdir(os.path.join(d, "backup")); self.assertEqual(len(files), 1)
        self.assertEqual(open(os.path.join(d, "backup", files[0]), "rb").read(), b"old")

    def test_diagnose_tools_with_stand_in_policy(self):
        """Exercises diagnose.py code paths with a hand-written linear 'model' (NOT a trained PPO)."""
        import diagnose
        class Lin:
            w = np.array([3.0, 0, 0, 0, 2.0, 0, 0])
            def predict(self, o, deterministic=True):
                o = np.atleast_2d(o); return np.clip(o @ self.w, -1, 1).reshape(-1, 1), None
        rng = np.random.default_rng(0); obs = rng.normal(0, 0.3, (2000, 7)).astype(np.float32)
        s = diagnose.feature_sensitivity(Lin(), obs, 0.05)
        self.assertGreater(s["dT"]["induced_u_std"], s["error"]["induced_u_std"])   # dT noise dominates
        self.assertEqual(s["target"]["induced_u_std"], 0.0)

    def test_convergence_report(self):
        import diagnose
        d = tempfile.mkdtemp(); p = os.path.join(d, "m.csv")
        rr = np.random.default_rng(0)
        with open(p, "w") as f:
            f.write('#{"t":0}\nr,l,t\n' + "\n".join(f"{-300 + 0.5 * i + rr.normal(0, 3)},400,{i}" for i in range(400)))
        self.assertTrue(diagnose.training_convergence(p)["still_improving(>2 s.e.)"])
        with open(p, "w") as f:
            f.write('#{"t":0}\nr,l,t\n' + "\n".join(f"{-100 + rr.normal(0, 3)},400,{i}" for i in range(400)))
        self.assertFalse(diagnose.training_convergence(p)["still_improving(>2 s.e.)"])


if __name__ == "__main__":
    unittest.main()
