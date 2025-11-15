import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.gridEnv import GridEnv

# Load trained model
model = PPO.load("models/ppo_grid_model")

# Create environment
env = GridEnv(battery_power_kw=100, render_every=0)
obs, _ = env.reset()

# Lists to store data
soc_list = []
load_list = []
solar_list = []
net_grid_list = []
action_list = []

done = False

while not done:
    # Predict action
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

    # Record data
    soc_list.append(env.soc)
    load_list.append(env.load[env.t-1])
    solar_list.append(env.solar[env.t-1])
    net_grid_list.append(env.load[env.t-1] - env.solar[env.t-1] + float(action[0]*env.max_power))
    action_list.append(float(action[0]*env.max_power))

# ---------------------------
# Plot results
# ---------------------------
t = np.arange(len(soc_list))

plt.figure(figsize=(12,6))
plt.plot(t, soc_list, label="Battery SoC")
plt.title("Battery SoC Over Time")
plt.ylabel("SoC (0-1)")
plt.xlabel("Timestep")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(t, load_list, label="Load")
plt.plot(t, solar_list, label="Solar")
plt.plot(t, net_grid_list, label="Net Grid Import")
plt.title("Load, Solar, and Net Grid Import")
plt.ylabel("Normalized kW")
plt.xlabel("Timestep")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(t, action_list, label="Battery Action (kW)")
plt.title("Agent Battery Actions")
plt.ylabel("Charge/Discharge Power (kW)")
plt.xlabel("Timestep")
plt.legend()
plt.grid(True)
plt.show()
