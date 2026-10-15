import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

model = PPO.load(sys.argv[1], device="cpu")
env = Monitor(gym.make("LunarLander-v3"))   # built-in reward only

scores, lengths = evaluate_policy(
    model, env, n_eval_episodes=20, deterministic=True, return_episode_rewards=True
)
print(f"Mean score:  {np.mean(scores):.1f} +/- {np.std(scores):.1f}")
print(f"Mean length: {np.mean(lengths):.0f} frames  (1000 = it never landed)")
print("SOLVED" if np.mean(scores) >= 200 else "NOT SOLVED")