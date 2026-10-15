import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset(seed=42)

for step in range(1000):
    action = env.action_space.sample()          # random: 0=nothing, 1=left, 2=main, 3=right
    obs, reward, terminated, truncated, info = env.step(action)
    if step % 10 == 0:
        print(np.round(obs, 2), "reward:", round(reward, 2))
    if terminated or truncated:
        print("--- episode over ---")
        obs, info = env.reset()

env.close()