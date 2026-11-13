"""Compare PID with one or more PPO models on IDENTICAL scenarios, initial conditions, disturbances and noise.
   python compare_models.py --models old=backups/v1/models/ppo_temperature.zip new=models/ppo_v2.zip
Each model uses the config saved next to it (<model>.yaml) if present, else config.yaml (v1 settings).
Outputs results/compare/: metrics_compare.csv (per case), summary_random.csv (held-out random scenarios), plots.
'return' is deliberately NOT compared: it depends on the reward weights, which differ between versions."""
import argparse, csv, os
import numpy as np
from thermal_model import load_config, random_scenario, rollout
from pid_controller import PIDController
from evaluate import compute_metrics, build_experiments
import visualization as viz

SIX = ["iae", "rmse", "overshoot_pct", "settling_time", "ss_error", "control_tv"]
EXTRA = ["control_var_ss", "dist_max_dev", "dist_recovery_time"]


def model_cfg(path, base):
    side = os.path.splitext(path)[0] + ".yaml"
    cfg = load_config(side) if os.path.exists(side) else base
    assert cfg["plant"] == base["plant"], f"plant differs for {path}: comparison would not be fair"
    return cfg


def nanmean(vals):
    v = [x for x in vals if x is not None]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (None, None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, metavar="LABEL=PATH")
    ap.add_argument("--random", type=int, default=50, help="number of held-out random scenarios")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    base = load_config(args.config)
    from ppo_agent import load_model, PPOController
    ctrls = {"PID": (PIDController.from_config(base), base)}
    for item in args.models:
        label, path = item.split("=", 1)
        c = model_cfg(path, base)
        ctrls[label] = (PPOController(load_model(path)), c)
    out = os.path.join(base["eval"]["results_dir"], "compare"); os.makedirs(out, exist_ok=True)

    # ---- fixed experiments (same scenario objects for every controller)
    rows, results = [], {}
    for exp, cases in build_experiments(base).items():
        results[exp] = []
        for label, sc in cases:
            runs = {n: rollout(c, sc, cfg) for n, (c, cfg) in ctrls.items()}
            mets = {n: compute_metrics(r, base) for n, r in runs.items()}   # metrics always with the SAME settings
            results[exp].append({"label": label, "scenario": sc, "runs": runs, "metrics": mets})
            for n, m in mets.items():
                rows.append({"experiment": exp, "case": label, "controller": n,
                             **{k: ("N/A" if m[k] is None else m[k]) for k in SIX + EXTRA},
                             "unsettled_segments": m["unsettled_segments"], "tripped": m["tripped"]})
    with open(os.path.join(out, "metrics_compare.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- held-out random scenarios (identical for all controllers)
    rng = np.random.default_rng(2024)
    scs = [random_scenario(rng, base) for _ in range(args.random)]
    summ = []
    print(f"\nHeld-out random scenarios (n={args.random}): mean +/- std (n valid)")
    print(f"{'controller':<10}" + "".join(f"{k:>24}" for k in SIX))
    for n, (c, cfg) in ctrls.items():
        ms = [compute_metrics(rollout(c, sc, cfg), base) for sc in scs]
        line, rec = f"{n:<10}", {"controller": n}
        for k in SIX:
            mu, sd, cnt = nanmean([m[k] for m in ms])
            rec[k + "_mean"], rec[k + "_std"], rec[k + "_n"] = mu, sd, cnt
            line += f"{'N/A' if mu is None else f'{mu:.3f}+/-{sd:.3f} ({cnt})':>24}"
        rec["trips"] = sum(m["tripped"] for m in ms)
        summ.append(rec); print(line + (f"   TRIPS={rec['trips']}" if rec["trips"] else ""))
    with open(os.path.join(out, "summary_random.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader()
        w.writerows([{k: ("N/A" if v is None else v) for k, v in r.items()} for r in summ])

    # ---- plots
    viz.plot_case(results["exp1_constant_target"][0], f"{out}/exp1_constant_target.png", "Exp 1: constant target")
    viz.plot_case(results["exp2_changing_target"][0], f"{out}/exp2_changing_target.png", "Exp 2: changing target")
    viz.plot_case(results["exp3_disturbance"][0], f"{out}/exp3_disturbance_response.png", "Exp 3: disturbances")
    viz.plot_multi(results["exp4_initial_temps"], f"{out}/exp4_initial_temperatures.png", "Exp 4: initial temperatures")
    viz.plot_multi(results["exp5_heating_capacity"], f"{out}/exp5_heating_capacity.png", "Exp 5: heating capacity")
    viz.plot_metric_bars(results, f"{out}/comparison_metrics.png")
    print("\nSaved to", out)


if __name__ == "__main__":
    main()
