"""Custom Gymnasium environment for the thermal plant.

Observation (7-D, see thermal_model.ObsBuilder): scaled error, temperature, target,
ambient, temperature change, clipped integral of error, previous action.
Action (1-D, [-1,1]): mapped to heater fraction u = (a+1)/2  -> 0..P_max (heating only).
Reward: -(w_error*|e| + w_effort*u + w_smooth*|du|); -safety_penalty on over-temperature trip.
Termination: T > T_safe_max.  Truncation: end of episode (episode_steps).
The disturbance is NOT observed by the agent (it must be inferred from the response).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from thermal_model import (ThermalPlant, make_obs_builder, random_scenario, compute_reward, load_config)


class TemperatureControlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, scenario=None):
        super().__init__()
        self.cfg = config or load_config()
        self.fixed_scenario = scenario
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.ob = make_obs_builder(self.cfg)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "scenario" in options:
            self.sc = options["scenario"]
        elif self.fixed_scenario is not None:
            self.sc = self.fixed_scenario
        else:
            self.sc = random_scenario(self.np_random, self.cfg)
        self.plant = ThermalPlant(self.cfg["plant"], self.sc.capacity_scale)
        self.T, self.k, self.u_prev = self.sc.T0, 0, 0.0
        self.ob.reset()
        obs = self._obs()
        return obs, {"T": self.T, "target": float(self.sc.target[0])}

    def _obs(self):
        sc, k = self.sc, self.k
        return self.ob.build(self.T + sc.noise[k], sc.target[k], sc.ambient[k])

    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        u = (a + 1.0) / 2.0
        sc, k = self.sc, self.k
        T_next = self.plant.step(self.T, u, sc.ambient[k], sc.disturbance[k])
        reward = compute_reward(T_next, sc.target[k + 1], u, self.u_prev, self.cfg["reward"])
        self.ob.commit(u)
        self.u_prev, self.T, self.k = u, T_next, k + 1
        terminated = bool(self.T > self.plant.T_safe_max)
        if terminated:
            reward -= self.cfg["reward"]["safety_penalty"]
        truncated = bool(self.k >= sc.n) and not terminated
        info = {"T": self.T, "target": float(sc.target[self.k]), "u": u}
        return self._obs(), float(reward), terminated, truncated, info
