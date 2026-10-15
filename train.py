import argparse
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from rewards import WRAPPERS

p = argparse.ArgumentParser()
p.add_argument("--reward", default="none", choices=WRAPPERS.keys())
p.add_argument("--steps", type=int, default=1_000_000)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

# 16 copies of the game run side by side, which makes training faster.
env = make_vec_env(
    "LunarLander-v3",
    n_envs=16,
    seed=args.seed,
    wrapper_class=WRAPPERS[args.reward],
)

# Settings from the Hugging Face Deep RL course, Unit 1
model = PPO(
    "MlpPolicy", env,
    n_steps=1024, batch_size=64, n_epochs=4,
    gamma=0.999, gae_lambda=0.98, ent_coef=0.01,
    seed=args.seed, device="cpu", verbose=1,
    tensorboard_log="runs",
)

name = f"ppo_{args.reward}_s{args.seed}"
start = time.time()
model.learn(total_timesteps=args.steps, tb_log_name=name)
print(f"Trained in {(time.time() - start) / 60:.1f} minutes")
model.save(f"models/{name}")
print(f"Saved models/{name}.zip")