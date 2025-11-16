import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.gridEnvV2 import GridEnvV2
import os  # <- added for folder creation

# ---- Load Env & Model ----
env = GridEnvV2()
model = PPO.load("models/ppo_grid_model_v2")

obs, _ = env.reset()
battery_trace = []
netload_trace = []
load_trace = []
solar_trace = []

done = False
step = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # logging
    load = env.load[env.current_step]
    solar = env.solar[env.current_step]
    battery = env.battery_soc

    net_load = load - solar
    if action == 1:  # charge
        net_load += env.charge_rate
    elif action == -1 or action == 2:  # discharge
        net_load -= env.discharge_rate

    battery_trace.append(battery)
    netload_trace.append(net_load)
    load_trace.append(load)
    solar_trace.append(solar)

    done = terminated
    step += 1

# ---- Create folder for saving outputs ----
os.makedirs("evalOutputs", exist_ok=True)

# ---- Save evaluation outputs ----
np.save("evalOutputs/original_load.npy", np.array(load_trace))
np.save("evalOutputs/optimized_load.npy", np.array(netload_trace))
np.save("evalOutputs/battery_actions.npy", np.array(battery_trace))

# If you tracked rewards in env/step, replace this line with actual values
rewards = np.zeros(len(load_trace))  # placeholder if rewards not collected
np.save("evalOutputs/rewards.npy", rewards)

print("\nEvaluation completed. Outputs saved to evalOutputs folder.")

# ---- Optional: Quick plots ----
plt.figure(figsize=(12, 4))
plt.title("Battery State-of-Charge (SoC)")
plt.plot(battery_trace)
plt.ylabel("SoC (kWh)")
plt.xlabel("Time Step")
plt.grid()
plt.show()

plt.figure(figsize=(12, 4))
plt.title("Net Load After RL Control")
plt.plot(netload_trace)
plt.ylabel("kW")
plt.xlabel("Time Step")
plt.grid()
plt.show()

plt.figure(figsize=(12, 4))
plt.title("Original Load vs Solar")
plt.plot(load_trace, label="Load")
plt.plot(solar_trace, label="Solar")
plt.legend()
plt.grid()
plt.show()
