"""Optional Streamlit demo.   streamlit run app.py
Lets you choose scenario parameters and compare PID (always) with PPO (if models/ppo_temperature.zip exists)."""
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from thermal_model import load_config, make_scenario, rollout
from pid_controller import PIDController
from evaluate import compute_metrics

cfg = load_config()
st.title("RL vs PID - Industrial Temperature Control (simulation)")
T0 = st.sidebar.slider("Initial temperature [°C]", 15, 60, 25)
tgt1 = st.sidebar.slider("Target [°C]", 30, 55, 45)
tgt2 = st.sidebar.slider("Target after t=200 s [°C]", 30, 55, 45)
scale = st.sidebar.slider("Heater capacity scale", 0.6, 1.4, 1.0)
amb = st.sidebar.slider("Ambient [°C]", 10, 30, 20)
dist = st.sidebar.slider("Disturbance 250-350 s [W] (negative = heat loss)", -300, 300, 0)

sc = make_scenario(cfg, T0, [(0, tgt1), (200, tgt2)], n=500, dist=[(250, 350, float(dist))],
                   ambient=[(0, float(amb))], scale=scale, seed=7)
ctrls = {"PID": PIDController.from_config(cfg)}
mp = cfg["ppo"]["model_path"]
if os.path.exists(mp):
    from ppo_agent import load_model, PPOController
    ctrls["PPO"] = PPOController(load_model(mp))
else:
    st.warning("No trained PPO model found - showing PID only. Run `python train.py`.")

runs = {k: rollout(c, sc, cfg) for k, c in ctrls.items()}
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
first = next(iter(runs.values()))
ax[0].plot(first["t"], first["target"], "k--", label="target")
for k, r in runs.items():
    ax[0].plot(r["t"], r["T"], label=k); ax[1].plot(r["t"], r["u"], label=k)
ax[0].set_ylabel("Temperature [°C]"); ax[1].set_ylabel("Heater fraction"); ax[1].set_xlabel("Time [s]")
ax[0].legend(); [a.grid(alpha=.3) for a in ax]
st.pyplot(fig)
st.table({k: {m: round(v, 3) for m, v in compute_metrics(r, cfg).items() if isinstance(v, float)}
          for k, r in runs.items()})
