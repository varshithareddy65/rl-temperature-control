import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "results/compare/summary_random.csv"
OUTPUT_PATH = "results/compare/heldout_metrics.png"

with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

controllers = [row["controller"] for row in rows]

metrics = [
    ("iae", "IAE (lower is better)"),
    ("rmse", "RMSE (lower is better)"),
    ("overshoot_pct", "Overshoot (%)"),
    ("settling_time", "Settling time (s)"),
    ("ss_error", "Steady-state error"),
    ("control_tv", "Control variation"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, (key, title) in zip(axes, metrics):
    means = [float(row[f"{key}_mean"]) for row in rows]
    stds = [float(row[f"{key}_std"]) for row in rows]

    bars = ax.bar(
        controllers,
        means,
        yerr=stds,
        capsize=4
    )

    ax.set_title(title)
    ax.set_ylabel("Mean ± standard deviation")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

fig.suptitle(
    "PID vs PPO Versions: 50 Held-out Random Scenarios",
    fontsize=15
)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=160)
plt.close(fig)

print(f"Chart saved to {OUTPUT_PATH}")
