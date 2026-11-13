"""Train PPO from scratch.
   python train.py                                   # v1 settings from config.yaml
   python train.py --tag v2 --set reward.w_smooth=0.3 --seed 0
Existing model/monitor files at the target paths are ALWAYS copied to models/backup/ first.
The exact config used is saved next to the model (models/ppo_<tag>.yaml) so evaluation uses matching settings."""
import argparse, os, shutil, time
import yaml
from thermal_model import load_config


def backup_if_exists(path, backup_dir):
    if os.path.exists(path):
        os.makedirs(backup_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(os.path.basename(path))
        dst = os.path.join(backup_dir, f"{base}_{stamp}{ext}")
        shutil.copy2(path, dst)
        print(f"Backed up {path} -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tag", default=None, help="writes models/ppo_<tag>.zip and logs/monitor_<tag>.csv")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="config overrides, e.g. reward.w_smooth=0.3")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    p = cfg["ppo"]
    if args.seed is not None:
        p["seed"] = args.seed
    if args.tag:
        p["model_path"] = f"models/ppo_{args.tag}.zip"
        p["monitor_path"] = f"logs/monitor_{args.tag}.csv"

    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecMonitor
    from stable_baselines3.common.utils import set_random_seed
    from environment import TemperatureControlEnv
    from ppo_agent import build_model

    os.makedirs(os.path.dirname(p["model_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(p["monitor_path"]), exist_ok=True)
    for f in (p["model_path"], p["monitor_path"]):
        backup_if_exists(f, os.path.join(os.path.dirname(f), "backup"))
        if f == p["monitor_path"] and os.path.exists(f):
            os.remove(f)                      # VecMonitor would otherwise overwrite/append confusingly
    set_random_seed(p["seed"])
    venv = make_vec_env(lambda: TemperatureControlEnv(cfg), n_envs=p["n_envs"], seed=p["seed"])
    venv = VecMonitor(venv, filename=p["monitor_path"])
    model = build_model(venv, cfg)
    model.learn(total_timesteps=args.timesteps or p["total_timesteps"], progress_bar=False)
    model.save(p["model_path"])
    side = os.path.splitext(p["model_path"])[0] + ".yaml"
    with open(side, "w") as f:
        yaml.safe_dump(cfg, f)
    print("Saved model:", p["model_path"], "| config used:", side)


if __name__ == "__main__":
    main()
