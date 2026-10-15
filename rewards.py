import gymnasium as gym


class TimePenalty(gym.Wrapper):
    """Charge a small fee every frame so taking too long costs something."""

    def __init__(self, env, fee=0.05):
        super().__init__(env)
        self.fee = fee

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward - self.fee, terminated, truncated, info


def is_calm(obs):
    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = obs
    return abs(angle) < 0.2 and abs(vx) < 0.2 and abs(vy) < 0.2


class HoverBonus(gym.Wrapper):
    """BROKEN ON PURPOSE: +1 every frame the rocket is upright and slow.
    Hovering for 1000 frames pays more than landing, so it learns to hover."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if is_calm(obs):
            reward += 1.0
        return obs, reward, terminated, truncated, info


class FixedBonus(gym.Wrapper):
    """THE FIX: pay for *getting* calmer (change between frames), not for
    *being* calm, and charge a fee every frame so waiting costs something."""

    def __init__(self, env, scale=10.0, fee=0.05):
        super().__init__(env)
        self.scale = scale
        self.fee = fee
        self.prev = 0.0

    @staticmethod
    def calmness(obs):
        # higher is better: less tilt, less speed
        return -(abs(obs[4]) + abs(obs[2]) + abs(obs[3]))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev = self.calmness(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        now = self.calmness(obs)
        reward += self.scale * (now - self.prev) - self.fee
        self.prev = now
        return obs, reward, terminated, truncated, info


WRAPPERS = {
    "none": None,
    "time": TimePenalty,
    "hover": HoverBonus,
    "fixed": FixedBonus,
}