import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class GridEnv(gym.Env):
    """
    RL Grid Environment (improved)
    - Uses normalized load, solar, price
    - Battery physics and economic reward
    - Encourages battery usage
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path="data/dataset.csv",
        battery_capacity_kwh=100.0,
        battery_power_kw=100.0,  # increased for stronger effect
        efficiency=0.92,
        timestep_hours=1.0,
        render_every=0  # 0 = disable printing
    ):
        super().__init__()

        self.render_every = render_every

        # Load dataset
        df = pd.read_csv(data_path)

        # Normalize between 0–1
        self.load = df["load"].values.astype(np.float32) / df["load"].max()
        self.solar = df["solar"].values.astype(np.float32) / df["solar"].max()
        self.price = df["price"].values.astype(np.float32) / df["price"].max()
        self.horizon = len(df)

        # Battery parameters
        self.capacity = battery_capacity_kwh
        self.max_power = battery_power_kw
        self.eff = efficiency
        self.dt = timestep_hours
        self.soc = 0.5
        self.t = 0

        # Observation space: [load, solar, price, soc]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Action space: -1 = discharge, +1 = charge
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    # Observation vector
    def _get_obs(self):
        return np.array([
            self.load[self.t],
            self.solar[self.t],
            self.price[self.t],
            self.soc
        ], dtype=np.float32)

    # Reset environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.soc = 0.5
        return self._get_obs(), {}

    # Step function
    def step(self, action):
        # Scale action to real power
        power_cmd = float(action[0]) * self.max_power

        # Battery charging/discharging physics
        if power_cmd >= 0:  # charging
            energy_change = power_cmd * self.dt * self.eff
        else:  # discharging
            energy_change = power_cmd * self.dt / self.eff

        # Update SoC
        self.soc = np.clip(self.soc + energy_change / self.capacity, 0.0, 1.0)

        # Grid economics
        load = self.load[self.t]
        solar = self.solar[self.t]
        price = self.price[self.t]
        net_grid = load - solar + power_cmd
        cost = net_grid * price * self.dt

        # Reward: negative cost + soc penalty + incentive for useful battery actions
        soc_penalty = -3 * (self.soc < 0.05 or self.soc > 0.95)
        action_reward = 0.5 * power_cmd * np.sign(load - solar)
        reward = -cost + soc_penalty + action_reward

        # Step forward
        self.t += 1
        done = self.t >= self.horizon - 1

        # Optional render
        if self.render_every and self.t % self.render_every == 0:
            print(f"t={self.t}, load={load:.3f}, solar={solar:.3f}, soc={self.soc:.3f}")

        return self._get_obs(), reward, done, False, {}

    # Render for debugging
    def render(self, mode="human"):
        print(f"t={self.t}, load={self.load[self.t]:.3f}, solar={self.solar[self.t]:.3f}, soc={self.soc:.3f}")
