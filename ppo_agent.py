"""PPO (Stable-Baselines3) construction/loading and a controller wrapper for evaluation."""
import numpy as np


def build_model(env, cfg):
    import torch
    from stable_baselines3 import PPO
    p = cfg["ppo"]
    return PPO(
        "MlpPolicy", env,
        n_steps=p["n_steps"], batch_size=p["batch_size"], n_epochs=p["n_epochs"],
        learning_rate=p["learning_rate"], gamma=p["gamma"], gae_lambda=p["gae_lambda"],
        clip_range=p["clip_range"], ent_coef=p["ent_coef"], seed=p["seed"], device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=list(p["net_arch"]), vf=list(p["net_arch"])),
                           activation_fn=torch.nn.Tanh),
        verbose=1,
    )


def load_model(path):
    from stable_baselines3 import PPO
    return PPO.load(path, device="cpu")


class PPOController:
    """Deterministic policy (mean action) -> heater fraction in [0,1]."""
    def __init__(self, model):
        self.model = model

    def reset(self):
        pass

    def act(self, obs, target, T_meas):
        a, _ = self.model.predict(obs, deterministic=True)
        return float((np.clip(a, -1, 1).reshape(-1)[0] + 1.0) / 2.0)
