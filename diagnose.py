"""Diagnose a trained PPO model.   python diagnose.py models/ppo_temperature.zip [--config models/ppo_v2.yaml]
A. Is training converged?   B. Which observation feature drives action jitter?
C. Is the policy stable / reproducible?   D. Compare jitter with PID.
Reports measurements and the thresholds used; it does not decide for you."""
import argparse, csv, json, os
import numpy as np
from thermal_model import load_config, random_scenario, rollout, make_scenario
from pid_controller import PIDController
from evaluate import compute_metrics

FEATURES = ["error", "T", "target", "ambient", "dT", "e_int", "prev_u"]


class Recorder:
    """Wraps a controller and records every observation it sees."""
    def __init__(self, c): self.c, self.obs = c, []
    def reset(self): self.c.reset(); self.obs = []
    def act(self, obs, target, T): self.obs.append(obs.copy()); return self.c.act(obs, target, T)


def training_convergence(monitor_path):
    with open(monitor_path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    hdr, d = rows[0], np.array(rows[1:], dtype=float)
    r, l = d[:, hdr.index("r")], d[:, hdr.index("l")]
    n = len(r); k = max(n // 5, 5)
    last, prev = r[-k:], r[-2 * k:-k]
    se = np.sqrt(last.var(ddof=1) / len(last) + prev.var(ddof=1) / len(prev))
    return {"episodes": n, "timesteps": int(l.sum()), "mean_return_prev_20pct": float(prev.mean()),
            "mean_return_last_20pct": float(last.mean()), "improvement": float(last.mean() - prev.mean()),
            "std_error_of_difference": float(se),
            "still_improving(>2 s.e.)": bool(last.mean() - prev.mean() > 2 * se),
            "std_return_last_20pct": float(last.std())}


def feature_sensitivity(model, obs, sensor_std, n_rep=20, seed=0):
    """Action std (in heater fraction u) caused by realistic noise injected into ONE feature at a time."""
    rng = np.random.default_rng(seed)
    noise_sd = {"error": sensor_std / 10, "T": sensor_std / 20, "target": 0.0, "ambient": 0.0,
                "dT": np.sqrt(2) * sensor_std / 0.5, "e_int": 0.0, "prev_u": 0.0}
    base = (np.clip(model.predict(obs, deterministic=True)[0].reshape(-1), -1, 1) + 1) / 2
    out = {}
    for i, name in enumerate(FEATURES):
        devs = []
        for _ in range(n_rep):
            o = obs.copy(); o[:, i] += rng.normal(0, noise_sd[name], len(o))
            u = (np.clip(model.predict(o, deterministic=True)[0].reshape(-1), -1, 1) + 1) / 2
            devs.append(u - base)
        out[name] = {"input_noise_sd": float(noise_sd[name]), "induced_u_std": float(np.std(devs))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--config", default=None)
    ap.add_argument("--monitor", default=None); ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    side = os.path.splitext(a.model)[0] + ".yaml"
    cfg = load_config(a.config or (side if os.path.exists(side) else "config.yaml"))
    from ppo_agent import load_model, PPOController
    model = load_model(a.model); ppo = PPOController(model)
    pid = PIDController.from_config(cfg); sd = cfg["plant"]["sensor_noise_std"]
    report = {}

    mon = a.monitor or cfg["ppo"]["monitor_path"]
    if os.path.exists(mon):
        report["A_training"] = training_convergence(mon)
    else:
        report["A_training"] = f"no monitor log at {mon}"

    rng = np.random.default_rng(777)
    scs = [random_scenario(rng, cfg) for _ in range(a.n)]
    rec = Recorder(ppo); allobs, ppo_m, pid_m = [], [], []
    for sc in scs:
        ppo_m.append(compute_metrics(rollout(rec, sc, cfg), cfg)); allobs.append(np.array(rec.obs))
        pid_m.append(compute_metrics(rollout(pid, sc, cfg), cfg))
    obs = np.concatenate(allobs)
    report["B_feature_sensitivity"] = feature_sensitivity(model, obs, sd)

    g = lambda ms, k: float(np.mean([m[k] for m in ms]))
    report["D_jitter_vs_pid"] = {k: {"PPO": g(ppo_m, k), "PID": g(pid_m, k)}
                                 for k in ("control_var", "control_var_ss", "control_tv", "ss_error", "iae")}

    # C. stability/reproducibility: same scenario, 20 different noise realisations; check chatter structure
    base_sc = make_scenario(cfg, 25, [(0, 45)], n=400, seed=0)
    iaes, ss_u_std, flips = [], [], []
    for s in range(20):
        sc = make_scenario(cfg, 25, [(0, 45)], n=400, seed=100 + s)
        r = rollout(ppo, sc, cfg); iaes.append(compute_metrics(r, cfg)["iae"])
        du = np.diff(r["u"][-100:]); ss_u_std.append(r["u"][-100:].std())
        flips.append(float(np.mean(np.sign(du[1:]) != np.sign(du[:-1]))))
    report["C_stability"] = {"iae_mean": float(np.mean(iaes)), "iae_std_over_noise": float(np.std(iaes)),
                             "steady_u_std_last100": float(np.mean(ss_u_std)),
                             "sign_flip_fraction_of_du (white-noise chatter ~0.67; slow oscillation <<0.5)": float(np.mean(flips)),
                             "any_safety_trip": bool(any(m["tripped"] for m in ppo_m))}
    print(json.dumps(report, indent=2))
    os.makedirs(cfg["eval"]["results_dir"], exist_ok=True)
    with open(os.path.join(cfg["eval"]["results_dir"], f"diagnosis_{os.path.basename(a.model)}.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
