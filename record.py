import sys

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

model_path, label = sys.argv[1], sys.argv[2]
model = PPO.load(model_path, device="cpu")

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", name_prefix=label,
                  episode_trigger=lambda ep: True)

for ep in range(3):
    obs, info = env.reset(seed=ep)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
env.close()
print("Videos saved in videos/")