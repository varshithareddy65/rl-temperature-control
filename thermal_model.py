"""Gym-free core: thermal plant, scenarios, observation, reward and a rollout loop.

Physical model (lumped first-order thermal system):
    C * dT/dt = P_max * s * u  -  (T - T_amb) / R  +  d(t)
 =>  tau * dT/dt = -(T - T_amb) + R * (P_max * s * u + d),     tau = R*C
with u in [0,1] (heater fraction), s = capacity scale, d = disturbance heat [W].
With u, T_amb, d held constant over one step the exact solution is used:
    T[k+1] = T_inf + (T[k] - T_inf) * exp(-dt/tau),  T_inf = T_amb + R*(P_max*s*u + d)
"""
from dataclasses import dataclass
import numpy as np
import yaml


def load_config(path="config.yaml", overrides=None):
    """Load YAML; overrides = ["reward.w_smooth=0.3", ...] (dotted keys, YAML-parsed values)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


class ThermalPlant:
    def __init__(self, cfg_plant, capacity_scale=1.0):
        self.dt = cfg_plant["dt"]
        self.C = cfg_plant["C"]
        self.R = cfg_plant["R"]
        self.P_max = cfg_plant["P_max"] * capacity_scale
        self.T_safe_max = cfg_plant["T_safe_max"]
        self.tau = self.R * self.C
        self._a = np.exp(-self.dt / self.tau)

    def step(self, T, u, T_amb, d=0.0):
        u = float(np.clip(u, 0.0, 1.0))
        T_inf = T_amb + self.R * (self.P_max * u + d)
        return T_inf + (T - T_inf) * self._a


@dataclass
class Scenario:
    """Everything that makes an episode reproducible (same for PID and PPO)."""
    T0: float
    target: np.ndarray        # length n+1
    ambient: np.ndarray       # length n+1
    disturbance: np.ndarray   # length n+1, watts (+ = extra heat, - = heat loss)
    noise: np.ndarray         # length n+1, sensor noise
    capacity_scale: float = 1.0

    @property
    def n(self):
        return len(self.target) - 1


def piecewise(n, points):
    """points = [(start_index, value), ...] -> array of length n+1."""
    out = np.zeros(n + 1)
    pts = sorted(points)
    for i, (k, v) in enumerate(pts):
        end = pts[i + 1][0] if i + 1 < len(pts) else n + 1
        out[k:end] = v
    return out


def random_scenario(rng, cfg):
    e, p = cfg["env"], cfg["plant"]
    n = e["episode_steps"]
    scale = rng.uniform(*e["capacity_scale_range"])
    amb0 = rng.uniform(*e["ambient_range"])
    ambient = np.full(n + 1, amb0)
    if rng.random() < e["ambient_step_prob"]:
        ambient[rng.integers(100, 300):] = amb0 + rng.uniform(-1, 1) * e["ambient_step_max"]
    k_max = p["R"] * p["P_max"] * scale
    hi = ambient.min() + e["feasibility_margin"] * k_max
    lo = ambient.max() + 3.0
    hi = max(hi, lo + 1.0)
    a, b = max(lo, e["target_range"][0]), min(hi, e["target_range"][1])
    if a >= b:
        a, b = lo, hi
    n_changes = rng.integers(0, e["max_target_changes"] + 1)
    times = sorted(rng.choice(np.arange(60, n - 60), size=n_changes, replace=False)) if n_changes else []
    pts = [(0, rng.uniform(a, b))] + [(int(k), rng.uniform(a, b)) for k in times]
    dist = np.zeros(n + 1)
    if rng.random() < e["disturbance_prob"]:
        k0, dur = int(rng.integers(50, 300)), int(rng.integers(30, 150))
        dist[k0:k0 + dur] = rng.uniform(-1, 1) * e["disturbance_max_W"]
    return Scenario(T0=rng.uniform(*e["init_temp_range"]), target=piecewise(n, pts),
                    ambient=ambient, disturbance=dist,
                    noise=rng.normal(0.0, p["sensor_noise_std"], n + 1), capacity_scale=scale)


def make_scenario(cfg, T0, targets, n=400, dist=(), ambient=((0, 20.0),), scale=1.0, seed=0):
    """Hand-built deterministic scenario. dist = [(start, end, watts), ...]."""
    d = np.zeros(n + 1)
    for s, e_, w in dist:
        d[s:e_] = w
    noise = np.random.default_rng(seed).normal(0.0, cfg["plant"]["sensor_noise_std"], n + 1)
    return Scenario(T0, piecewise(n, targets), piecewise(n, ambient), d, noise, scale)


class ObsBuilder:
    """Builds the 7-D observation. Shared by the Gym env and the rollout loop."""
    def __init__(self, dt, dT_alpha=1.0):
        # dT_alpha=1.0 -> raw temperature difference (original behaviour, keeps old models valid).
        # dT_alpha<1  -> exponential moving average of the difference (reduces sensor-noise amplification).
        self.dt, self.alpha = dt, dT_alpha
        self.reset()

    def reset(self):
        self.T_prev = None
        self.dT_f = 0.0
        self.e_int = 0.0
        self.prev_u = 0.0

    def build(self, T_meas, target, T_amb):
        e = target - T_meas
        self.e_int = float(np.clip(self.e_int + e * self.dt / 50.0, -1.0, 1.0))
        raw = 0.0 if self.T_prev is None else T_meas - self.T_prev
        self.dT_f = self.dT_f + self.alpha * (raw - self.dT_f)
        dT = self.dT_f
        self.T_prev = T_meas
        return np.array([e / 10.0, (T_meas - 40.0) / 20.0, (target - 40.0) / 20.0,
                         (T_amb - 20.0) / 10.0, dT / 0.5, self.e_int, 2 * self.prev_u - 1],
                        dtype=np.float32)

    def commit(self, u):
        self.prev_u = float(u)


def reward_components(T, target, u, u_prev, cfg_r):
    """(error_term, effort_term, smoothness_term), all >= 0 costs."""
    return (cfg_r["w_error"] * abs(target - T), cfg_r["w_effort"] * u,
            cfg_r["w_smooth"] * abs(u - u_prev))


def compute_reward(T, target, u, u_prev, cfg_r):
    return -sum(reward_components(T, target, u, u_prev, cfg_r))


def make_obs_builder(cfg):
    return ObsBuilder(cfg["plant"]["dt"], cfg.get("obs", {}).get("dT_alpha", 1.0))


def rollout(controller, sc, cfg):
    """Closed-loop simulation. controller.act(obs, target, T_meas) -> u in [0,1]."""
    plant = ThermalPlant(cfg["plant"], sc.capacity_scale)
    ob = make_obs_builder(cfg)
    controller.reset()
    T, u_prev = sc.T0, 0.0
    rec = {k: [] for k in ("T", "target", "u", "d", "ambient", "reward")}
    tripped = False
    for k in range(sc.n):
        Tm = T + sc.noise[k]
        obs = ob.build(Tm, sc.target[k], sc.ambient[k])
        u = float(np.clip(controller.act(obs, sc.target[k], Tm), 0.0, 1.0))
        T_next = plant.step(T, u, sc.ambient[k], sc.disturbance[k])
        r = compute_reward(T_next, sc.target[k + 1], u, u_prev, cfg["reward"])
        for key, val in zip(rec, (T, sc.target[k], u, sc.disturbance[k], sc.ambient[k], r)):
            rec[key].append(val)
        ob.commit(u)
        u_prev, T = u, T_next
        if T > plant.T_safe_max:
            tripped = True
            break
    out = {k: np.array(v) for k, v in rec.items()}
    out["t"] = np.arange(len(out["T"])) * cfg["plant"]["dt"]
    out["tripped"] = tripped
    return out
