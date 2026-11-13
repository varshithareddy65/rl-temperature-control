"""Metrics + the five experiments.   python evaluate.py [--model models/ppo_temperature.zip]
PID is always evaluated. PPO is evaluated only if a trained model file exists."""
import argparse, csv, json, os
import numpy as np
from thermal_model import load_config, make_scenario, rollout
from pid_controller import PIDController


# ---------------------------------------------------------------- metrics
def _segments(target):
    idx = [0] + [k for k in range(1, len(target)) if target[k] != target[k - 1]] + [len(target)]
    return list(zip(idx[:-1], idx[1:]))


def compute_metrics(res, cfg):
    dt, band = cfg["plant"]["dt"], cfg["eval"]["settle_band"]
    T, tgt, u = res["T"], res["target"], res["u"]
    e = tgt - T
    m = {"rmse": float(np.sqrt(np.mean(e ** 2))), "iae": float(np.sum(np.abs(e)) * dt),
         "ss_error": float(np.mean(np.abs(e[-cfg["eval"]["ss_window"]:]))),
         "mean_u": float(np.mean(u)), "mean_u2": float(np.mean(u ** 2)),
         "control_tv": float(np.sum(np.abs(np.diff(u)))),                 # total variation over the episode
         "control_var": float(np.mean(np.abs(np.diff(u)))),               # mean |du| per step (length-independent)
         "control_var_ss": float(np.mean(np.abs(np.diff(u[-cfg["eval"]["ss_window"]:])))),  # chatter at steady state
         "return": float(np.sum(res["reward"])), "tripped": bool(res["tripped"])}
    os_list, ts_list, unsettled = [], [], 0
    T0 = T[0]
    for s, f in _segments(tgt):
        seg_T, target = T[s:f], tgt[s]
        start = T0 if s == 0 else T[s - 1]
        step = target - start
        if abs(step) < 1.0 or len(seg_T) < 5:
            continue
        sign = np.sign(step)
        peak = seg_T.max() - target if sign > 0 else target - seg_T.min()
        os_list.append(max(0.0, peak) / abs(step) * 100.0)
        out = np.where(np.abs(seg_T - target) > band)[0]
        if len(out) == 0:
            ts_list.append(0.0)
        elif out[-1] == len(seg_T) - 1:
            ts_list.append(len(seg_T) * dt); unsettled += 1
        else:
            ts_list.append((out[-1] + 1) * dt)
    # None == "not applicable": no setpoint step >= 1 K in this scenario (e.g. run starts at the target)
    m["overshoot_pct"] = float(np.mean(os_list)) if os_list else None
    m["settling_time"] = float(np.mean(ts_list)) if ts_list else None
    m["unsettled_segments"] = unsettled
    # disturbance response: first change in heat disturbance or ambient
    d, amb = res["d"], res["ambient"]
    ev = [k for k in range(1, len(d)) if d[k] != d[k - 1] or amb[k] != amb[k - 1]]
    if ev:
        i0 = ev[0]; i1 = ev[1] if len(ev) > 1 else len(d)
        ea = np.abs(e[i0:i1])
        out = np.where(ea > band)[0]
        rec = 0.0 if len(out) == 0 else (len(ea) * dt if out[-1] == len(ea) - 1 else (out[-1] + 1) * dt)
        m["dist_max_dev"] = float(np.abs(e[i0:]).max()); m["dist_recovery_time"] = float(rec)
    else:  # no disturbance / ambient change in this scenario -> genuinely not applicable
        m["dist_max_dev"] = m["dist_recovery_time"] = None
    return m


# ---------------------------------------------------------------- experiments
def build_experiments(cfg):
    S = lambda *a, **k: make_scenario(cfg, *a, **k)
    return {
        "exp1_constant_target": [("T0=25, target=50", S(25, [(0, 50)], n=400, seed=1))],
        "exp2_changing_target": [("targets 40->55->35", S(30, [(0, 40), (150, 55), (300, 35)], n=500, seed=2))],
        "exp3_disturbance": [("-300 W loss @200-300s, ambient 20->10 @350s",
                              S(45, [(0, 45)], n=500, dist=[(200, 300, -300.0)],
                                ambient=[(0, 20.0), (350, 10.0)], seed=3))],
        "exp4_initial_temps": [(f"T0={t0}, target=45", S(t0, [(0, 45)], n=400, seed=4))
                               for t0 in (20, 35, 55)],
        "exp5_heating_capacity": [(f"capacity x{s}" + (" (outside PPO training range)" if s < 0.7 or s > 1.3 else ""),
                                   S(25, [(0, 40)], n=400, scale=s, seed=5)) for s in (0.6, 1.0, 1.4)],
    }


def run_all(controllers, cfg):
    results = {}
    for name, cases in build_experiments(cfg).items():
        results[name] = []
        for label, sc in cases:
            runs = {c: rollout(ctrl, sc, cfg) for c, ctrl in controllers.items()}
            mets = {c: compute_metrics(r, cfg) for c, r in runs.items()}
            results[name].append({"label": label, "scenario": sc, "runs": runs, "metrics": mets})
    return results


def fmt(v):
    return f"{'N/A':>12}" if v is None else f"{v:>12.3f}"


def print_and_save(results, cfg):
    cols = ["iae", "rmse", "overshoot_pct", "settling_time", "ss_error", "mean_u", "control_tv",
            "control_var_ss", "dist_max_dev", "dist_recovery_time"]
    rows = []
    hdr = f"{'case':<46}{'ctrl':<5}" + "".join(f"{c[:11]:>12}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for exp, cases in results.items():
        print(f"[{exp}]")
        for cs in cases:
            for c, m in cs["metrics"].items():
                print(f"{cs['label'][:45]:<46}{c:<5}" + "".join(fmt(m[k]) for k in cols)
                      + ("  UNSETTLED" if m["unsettled_segments"] else "") + ("  TRIPPED" if m["tripped"] else ""))
                rows.append({"experiment": exp, "case": cs["label"], "controller": c, **m})
    out = cfg["eval"]["results_dir"]; os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        w.writerows([{k: ("N/A" if v is None else v) for k, v in r.items()} for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    controllers = {"PID": PIDController.from_config(cfg)}
    model_path = args.model or cfg["ppo"]["model_path"]
    if os.path.exists(model_path):
        from ppo_agent import load_model, PPOController
        controllers["PPO"] = PPOController(load_model(model_path))
        print("Loaded PPO model:", model_path)
    else:
        print(f"!! No PPO model at {model_path} -> evaluating PID only. Run `python train.py` first.")
    results = run_all(controllers, cfg)
    print_and_save(results, cfg)
    import visualization as viz
    viz.make_all_plots(results, cfg)
    print("Plots + metrics saved to", cfg["eval"]["results_dir"])


if __name__ == "__main__":
    main()
