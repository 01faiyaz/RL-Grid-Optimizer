import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd


class GridEnv(gym.Env):
    """
    Pro-level Grid Environment for RL
    - real load, solar, price data (CSV)
    - physics-based battery model
    - economic reward function
    - supports 5–60 minute timesteps
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path="data/dataset.csv",
        battery_capacity_kwh=100.0,
        battery_power_kw=50.0,
        efficiency=0.92,
        timestep_hours=1.0,
    ):
        super().__init__()

        # -------------------------
        # Load dataset
        # -------------------------
        df = pd.read_csv(data_path)

        self.load = df["load"].values       # kW demand
        self.solar = df["solar"].values     # kW generation
        self.price = df["price"].values     # $/kWh

        self.horizon = len(df)

        # -------------------------
        # Battery parameters
        # -------------------------
        self.capacity = battery_capacity_kwh
        self.max_power = battery_power_kw
        self.eff = efficiency
        self.dt = timestep_hours

        # Battery state of charge (0–1)
        self.soc = 0.5

        # Time step
        self.t = 0

        # -------------------------
        # Observation + action space
        # -------------------------
        # obs = [load, solar, price, soc]
        high = np.array([1e4, 1e4, 5.0, 1.0], dtype=np.float32)
        low = np.array([-1e4, 0.0, 0.0, 0.0], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action = battery power (-1 to +1) scaled to battery_power_kw
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # ----------------------------------------------------
    # Helper — observation vector
    # ----------------------------------------------------
    def _get_obs(self):
        return np.array([
            self.load[self.t],
            self.solar[self.t],
            self.price[self.t],
            self.soc
        ], dtype=np.float32)

    # ----------------------------------------------------
    # Reset environment
    # ----------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.soc = 0.5
        return self._get_obs(), {}

    # ----------------------------------------------------
    # Step function (physics + energy economics)
    # ----------------------------------------------------
    def step(self, action):
        # Scale action to real power (kW)
        power_cmd = float(action[0]) * self.max_power

        # Positive = charge, Negative = discharge
        if power_cmd >= 0:
            energy_change = power_cmd * self.dt * self.eff
        else:
            energy_change = power_cmd * self.dt / self.eff

        # Update SoC
        new_soc = np.clip(self.soc + energy_change / self.capacity, 0.0, 1.0)
        self.soc = new_soc

        # -------------------------
        # Grid interaction
        # -------------------------
        load = self.load[self.t]
        solar = self.solar[self.t]
        price = self.price[self.t]

        # Net grid import after solar + battery
        net_grid = load - solar + power_cmd  # kW

        # Cost (positive import = cost, negative import = revenue)
        cost = net_grid * price * self.dt  # dollars

        # Penalty for stressing battery near 0% or 100%
        soc_penalty = -3 * (new_soc < 0.05 or new_soc > 0.95)

        # Reward = negative cost + penalty
        reward = -cost + soc_penalty

        # Advance time
        self.t += 1
        done = self.t >= self.horizon - 1

        return self._get_obs(), reward, done, False, {}

    # ----------------------------------------------------
    # Render (for debugging)
    # ----------------------------------------------------
    def render(self, mode="human"):
        print(f"t={self.t}, load={self.load[self.t]:.1f}, solar={self.solar[self.t]:.1f}, soc={self.soc:.3f}")
