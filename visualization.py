"""All plots. Uses a non-interactive backend so it runs headless."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"PID": "tab:blue", "PPO": "tab:red"}


def plot_case(case, path, title):
    """3 rows: temperature vs target, control action, disturbance & ambient."""
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.5]})
    first = next(iter(case["runs"].values()))
    ax[0].plot(first["t"], first["target"], "k--", label="target")
    for c, r in case["runs"].items():
        ax[0].plot(r["t"], r["T"], color=COLORS.get(c), label=f"{c} temperature")
        ax[1].plot(r["t"], r["u"], color=COLORS.get(c), label=c)
    ax[2].plot(first["t"], first["d"], color="tab:orange", label="disturbance heat [W]")
    ax2 = ax[2].twinx(); ax2.plot(first["t"], first["ambient"], color="tab:green", label="ambient [°C]")
    ax[0].set_ylabel("Temperature [°C]"); ax[1].set_ylabel("Heater fraction u")
    ax[2].set_ylabel("Disturbance [W]"); ax2.set_ylabel("Ambient [°C]"); ax[2].set_xlabel("Time [s]")
    ax[0].set_title(f"{title}\n{case['label']}"); ax[0].legend(); ax[1].legend(); ax[1].set_ylim(-0.05, 1.05)
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_multi(cases, path, title):
    n = len(cases)
    fig, ax = plt.subplots(2, n, figsize=(4.5 * n, 6), sharex=True, squeeze=False)
    for j, cs in enumerate(cases):
        first = next(iter(cs["runs"].values()))
        ax[0, j].plot(first["t"], first["target"], "k--", label="target")
        for c, r in cs["runs"].items():
            ax[0, j].plot(r["t"], r["T"], color=COLORS.get(c), label=c)
            ax[1, j].plot(r["t"], r["u"], color=COLORS.get(c), label=c)
        ax[0, j].set_title(cs["label"], fontsize=9); ax[1, j].set_xlabel("Time [s]")
        ax[1, j].set_ylim(-0.05, 1.05)
        for a in (ax[0, j], ax[1, j]): a.grid(alpha=0.3)
    ax[0, 0].set_ylabel("Temperature [°C]"); ax[1, 0].set_ylabel("Heater fraction u"); ax[0, 0].legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_metric_bars(results, path, keys=("iae", "overshoot_pct", "settling_time", "ss_error", "control_tv")):
    exps = ["exp1_constant_target", "exp2_changing_target", "exp3_disturbance"]
    ctrls = list(results[exps[0]][0]["metrics"])
    fig, ax = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 4))
    w = 0.8 / len(ctrls)
    for a, k in zip(ax, keys):
        for i, c in enumerate(ctrls):
            vals = [results[e][0]["metrics"][c][k] for e in exps]
            a.bar(np.arange(len(exps)) + i * w, np.array([0.0 if x is None else x for x in vals], dtype=float), w, color=COLORS.get(c), label=c)
        a.set_xticks(np.arange(len(exps)) + w * (len(ctrls) - 1) / 2); a.set_xticklabels(["const", "changing", "disturb."])
        a.set_title(k); a.grid(alpha=0.3, axis="y")
    ax[0].legend(); fig.suptitle("Controller comparison (lower is better for all shown metrics)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def read_monitor(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    hdr, data = rows[0], np.array(rows[1:], dtype=float)
    return data[:, hdr.index("r")], data[:, hdr.index("l")]


def plot_training(monitor_path, out_path, window=50):
    r, l = read_monitor(monitor_path)
    steps = np.cumsum(l)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, r, alpha=0.25, label="episode return")
    if len(r) >= window:
        ax.plot(steps[window - 1:], np.convolve(r, np.ones(window) / window, "valid"), label=f"{window}-episode mean")
    ax.set_xlabel("Training timesteps"); ax.set_ylabel("Episode return"); ax.set_title("PPO training reward")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def make_all_plots(results, cfg):
    out = cfg["eval"]["results_dir"]; os.makedirs(out, exist_ok=True)
    plot_case(results["exp1_constant_target"][0], f"{out}/exp1_constant_target.png", "Experiment 1: constant target")
    plot_case(results["exp2_changing_target"][0], f"{out}/exp2_changing_target.png", "Experiment 2: changing target")
    plot_case(results["exp3_disturbance"][0], f"{out}/exp3_disturbance_response.png", "Experiment 3: disturbances")
    plot_multi(results["exp4_initial_temps"], f"{out}/exp4_initial_temperatures.png", "Experiment 4: initial temperatures")
    plot_multi(results["exp5_heating_capacity"], f"{out}/exp5_heating_capacity.png", "Experiment 5: heating capacity")
    plot_metric_bars(results, f"{out}/comparison_metrics.png")
    mp = cfg["ppo"]["monitor_path"]
    if os.path.exists(mp):
        plot_training(mp, f"{out}/ppo_training_reward.png")
    else:
        print("(no training log found -> skipping ppo_training_reward.png)")
