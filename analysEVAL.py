import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# loading all values from outputs
data_folder = "evalOutputs"
original_load = np.load(f"{data_folder}/original_load.npy")
optimized_load = np.load(f"{data_folder}/optimized_load.npy")
battery_actions = np.load(f"{data_folder}/battery_actions.npy")
rewards = np.load(f"{data_folder}/rewards.npy")

time_steps = np.arange(len(original_load))

# making all the plots with matplotlib 
# mostly a grid with line plot
plt.figure(figsize=(12, 4))
plt.plot(original_load, label="Original Load")
plt.plot(optimized_load, label="RL Optimized Load")
plt.title("Duck Curve Comparison")
plt.xlabel("Time Step")
plt.ylabel("Load (kW)")
plt.legend()
plt.grid(True)
plt.savefig(f"{data_folder}/duck_curve_comparison.png")
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(battery_actions)
plt.title("Battery SoC")
plt.xlabel("Time Step")
plt.ylabel("kWh")
plt.grid(True)
plt.savefig(f"{data_folder}/battery_soc.png")
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(rewards)
plt.title("Reward per Step")
plt.xlabel("Time Step")
plt.ylabel("Reward")
plt.grid(True)
plt.savefig(f"{data_folder}/reward_curve.png")
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(optimized_load)
plt.title("Net Load After RL Control")
plt.xlabel("Time Step")
plt.ylabel("kW")
plt.grid(True)
plt.savefig(f"{data_folder}/net_load.png")
plt.show()

#calculating all the kpis
def calc_peak_reduction(original, optimized):
    return (original.max() - optimized.max()) / original.max() * 100

def calc_ramp_reduction(original, optimized):
    return (np.max(np.diff(original)) - np.max(np.diff(optimized))) / np.max(np.diff(original)) * 100

def calc_energy_utilization(original, optimized):
    curtailed_energy = np.sum(np.maximum(original - optimized, 0))
    total_available_energy = np.sum(original)
    return (1 - curtailed_energy / total_available_energy) * 100

peak_reduction = calc_peak_reduction(original_load, optimized_load)
ramp_reduction = calc_ramp_reduction(original_load, optimized_load)
renewable_utilization = calc_energy_utilization(original_load, optimized_load)
average_reward = np.mean(rewards)

kpis = {
    "Peak Reduction (%)": peak_reduction,
    "Ramp Reduction (%)": ramp_reduction,
    "Renewable Utilization (%)": renewable_utilization,
    "Average Reward": average_reward
}

print("\n=== RL Evaluation KPIs ===")
for k, v in kpis.items():
    print(f"{k}: {v:.2f}")

#making sure to save kpis to csv file
os.makedirs(data_folder, exist_ok=True)
df_kpis = pd.DataFrame([kpis])
df_kpis.to_csv(f"{data_folder}/kpis_summary.csv", index=False)
print(f"\nKPI summary saved to {data_folder}/kpis_summary.csv")
