import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class GridEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path="data/dataset.csv",
        battery_capacity_kwh=100.0,
        battery_power_kw=100.0,  # more = stronger
        efficiency=0.92,
        timestep_hours=1.0,
        render_every=0
    ):
        super().__init__()

        self.render_every = render_every

        # load dataset
        df = pd.read_csv(data_path)

        # mormalize between 0–1
        self.load = df["load"].values.astype(np.float32) / df["load"].max()
        self.solar = df["solar"].values.astype(np.float32) / df["solar"].max()
        self.price = df["price"].values.astype(np.float32) / df["price"].max()
        self.horizon = len(df)

        # param for battery
        self.capacity = battery_capacity_kwh
        self.max_power = battery_power_kw
        self.eff = efficiency
        self.dt = timestep_hours
        self.soc = 0.5
        self.t = 0
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # -1 = discharge, +1 = charge
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

    # resets env tom 0 for refresh
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.soc = 0.5
        return self._get_obs(), {}

    # step function
    def step(self, action):
        # scale action to real power
        power_cmd = float(action[0]) * self.max_power

        # Battery charging/discharging physics
        if power_cmd >= 0:  # charging
            energy_change = power_cmd * self.dt * self.eff
        else:  # discharging
            energy_change = power_cmd * self.dt / self.eff

        # SoC
        self.soc = np.clip(self.soc + energy_change / self.capacity, 0.0, 1.0)

        load = self.load[self.t]
        solar = self.solar[self.t]
        price = self.price[self.t]
        net_grid = load - solar + power_cmd
        cost = net_grid * price * self.dt

        # reward: negative cost + soc penalty + incentive for useful battery actions
        soc_penalty = -3 * (self.soc < 0.05 or self.soc > 0.95)
        action_reward = 0.5 * power_cmd * np.sign(load - solar)
        reward = -cost + soc_penalty + action_reward

        # go forward
        self.t += 1
        done = self.t >= self.horizon - 1

    # render this is for debugging it
    def render(self, mode="human"):
        print(f"t={self.t}, load={self.load[self.t]:.3f}, solar={self.solar[self.t]:.3f}, soc={self.soc:.3f}")
