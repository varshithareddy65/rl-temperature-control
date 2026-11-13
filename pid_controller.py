"""Discrete PID baseline.  u = Kp*e + Ki*integral(e) + Kd*d(filtered measurement)/dt, clipped to [0,1].
Ki = Kp/Ti.  Anti-windup: conditional integration (integrate only when not saturated
or when the error would pull the output back into range). Derivative acts on the
measurement (no kick on target changes) and is low-pass filtered.
"""
import numpy as np


class PIDController:
    def __init__(self, Kp, Ti, Kd=0.0, dt=1.0, deriv_filter_tau=5.0, u_min=0.0, u_max=1.0):
        self.Kp, self.Ki, self.Kd = Kp, Kp / Ti, Kd
        self.dt, self.tf, self.u_min, self.u_max = dt, deriv_filter_tau, u_min, u_max
        self.reset()

    @classmethod
    def from_config(cls, cfg):
        p = cfg["pid"]
        return cls(p["Kp"], p["Ti"], p["Kd"], cfg["plant"]["dt"], p["deriv_filter_tau"])

    def reset(self):
        self.integral, self.T_f, self.first = 0.0, None, True

    def act(self, obs, target, T_meas):
        e = target - T_meas
        if self.first:
            self.T_f, self.first = T_meas, False
        a = self.dt / (self.tf + self.dt)
        T_f_new = self.T_f + a * (T_meas - self.T_f)
        d_meas = (T_f_new - self.T_f) / self.dt
        self.T_f = T_f_new
        u_unsat = self.Kp * e + self.Ki * self.integral - self.Kd * d_meas
        u = float(np.clip(u_unsat, self.u_min, self.u_max))
        if u == u_unsat or (u_unsat > self.u_max and e < 0) or (u_unsat < self.u_min and e > 0):
            self.integral += e * self.dt
        return u


def tune(cfg, n_scenarios=12, seed=999):
    """Grid search on randomised scenarios (different seed from evaluation)."""
    from thermal_model import random_scenario, rollout
    from evaluate import compute_metrics
    rng = np.random.default_rng(seed)
    scs = [random_scenario(rng, cfg) for _ in range(n_scenarios)]
    best = None
    for Kp in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2):
        for Ti in (10, 15, 20, 30, 50, 80, 120, 200, 400):
            for Kd in (0.0, 10.0):
                c = PIDController(Kp, Ti, Kd, cfg["plant"]["dt"], cfg["pid"]["deriv_filter_tau"])
                score = 0.0
                for sc in scs:
                    m = compute_metrics(rollout(c, sc, cfg), cfg)
                    score += m["iae"] / sc.n + 2.0 * (m["overshoot_pct"] or 0.0) / 100 * 10 + 0.5 * m["control_tv"]
                if best is None or score < best[0]:
                    best = (score, Kp, Ti, Kd)
    return best


if __name__ == "__main__":
    from thermal_model import load_config
    cfg = load_config()
    score, Kp, Ti, Kd = tune(cfg)
    print(f"Best PID: Kp={Kp}, Ti={Ti}, Kd={Kd}  (score={score:.3f})")
    print("Copy these into config.yaml under `pid:`.")
