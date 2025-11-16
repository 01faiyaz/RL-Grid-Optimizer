import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from pathlib import Path

class GridEnvV2(gym.Env):
    """Improved RL environment with real Battery + Solar + Load system."""

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(GridEnvV2, self).__init__()

        # ---- LOAD DATA ----
        data_path = Path("data/dataset_normalized_test.csv")
        df = pd.read_csv(data_path)

        self.load = df["load"].values
        self.solar = df["solar"].values
        self.n = len(df)

        # ---- BATTERY SETTINGS ----
        self.max_battery = 100.0     # kWh
        self.min_battery = 0.0
        self.battery_soc = 50.0      # start half full
        self.charge_rate = 10.0      # kW per step
        self.discharge_rate = 10.0

        # ---- ACTION SPACE ----
        # -1 = discharge, 0 = idle, 1 = charge
        self.action_space = spaces.Discrete(3)

        # ---- OBSERVATION SPACE ----
        # load, solar, battery_soc
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 100], dtype=np.float32),
            dtype=np.float32
        )

        self.current_step = 0

    # ---------------------------------------------------------
    def step(self, action):
        load_t = self.load[self.current_step]
        solar_t = self.solar[self.current_step]

        # ---- BATTERY ACTION LOGIC ----
        if action == 1:  # charge battery
            self.battery_soc = min(self.max_battery, self.battery_soc + self.charge_rate)
        elif action == 0:  # idle
            pass
        elif action == -1 or action == 2:  # discharge (2 happens sometimes with PPO)
            self.battery_soc = max(self.min_battery, self.battery_soc - self.discharge_rate)

        # ---- NET LOAD AFTER BATTERY ----
        # if battery discharges → reduces load
        # if battery charges → increases net load
        net_load = load_t - solar_t

        if action == 1:  # charging → adds load
            net_load += self.charge_rate
        elif action == -1 or action == 2:  # discharging → subtracts load
            net_load -= self.discharge_rate

        # ---- REWARD FUNCTION ----
        # reward for reducing net load magnitude
        ramp_penalty = abs(net_load[t] - net_load[t-1])
        reward -= ramp_penalty

        # small penalty for very low or full battery
        if self.battery_soc < 5 or self.battery_soc > 95:
            reward -= 5

        # ---- NEXT STATE ----
        self.current_step += 1
        terminated = self.current_step >= self.n - 1

        obs = np.array([
            self.load[self.current_step],
            self.solar[self.current_step],
            self.battery_soc,
        ], dtype=np.float32)

        return obs, reward, terminated, False, {}

    # ---------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.battery_soc = 50.0

        initial_obs = np.array([
            self.load[0],
            self.solar[0],
            self.battery_soc
        ], dtype=np.float32)

        return initial_obs, {}

    # ---------------------------------------------------------
    def render(self):
        print(
            f"Step {self.current_step} | Load={self.load[self.current_step]:.3f} | "
            f"Solar={self.solar[self.current_step]:.3f} | Battery={self.battery_soc:.1f}"
        )
