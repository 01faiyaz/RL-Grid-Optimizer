import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# =========================
# Load evaluation outputs
# =========================
# Replace these with the paths to your evalV2 output files
data_folder = "evalOutputs"
original_load = np.load(os.path.join(data_folder, "original_load.npy"))   # baseline load MW
optimized_load = np.load(os.path.join(data_folder, "optimized_load.npy")) # RL optimized load MW
battery_actions = np.load(os.path.join(data_folder, "battery_actions.npy")) # +charge / -discharge MW
rewards = np.load(os.path.join(data_folder, "rewards.npy"))               # reward per timestep

time_steps = np.arange(len(original_load))  # x-axis for plotting

# =========================
# 1. Duck Curve Comparison
# =========================
plt.figure(figsize=(12,6))
plt.plot(time_steps, original_load, label="Original Load", color="red")
plt.plot(time_steps, optimized_load, label="RL Optimized Load", color="green")
plt.xlabel("Time Step (hour)")
plt.ylabel("Load (MW)")
plt.title("California Duck-Curve: Original vs RL Optimized")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(data_folder, "duck_curve_comparison.png"))
plt.show()

# =========================
# 2. Reward Curve
# =========================
plt.figure(figsize=(12,5))
plt.plot(rewards, label="Reward per Step", color="blue")
plt.xlabel("Time Step")
plt.ylabel("Reward")
plt.title("RL Training Reward Over Time")
plt.grid(True)
plt.savefig(os.path.join(data_folder, "reward_curve.png"))
plt.show()

# =========================
# 3. Battery Action Timeline
# =========================
plt.figure(figsize=(12,5))
plt.step(time_steps, battery_actions, where='post', label="Battery Actions (MW)")
plt.xlabel("Time Step")
plt.ylabel("Charge (+) / Discharge (-) (MW)")
plt.title("RL Agent Battery Action Timeline")
plt.grid(True)
plt.savefig(os.path.join(data_folder, "battery_actions.png"))
plt.show()

# =========================
# 4. KPI Calculations
# =========================
def calc_peak_reduction(original, optimized):
    return (original.max() - optimized.max()) / original.max() * 100

def calc_ramp_reduction(original, optimized):
    orig_ramp = np.max(np.diff(original))
    opt_ramp = np.max(np.diff(optimized))
    return (orig_ramp - opt_ramp) / orig_ramp * 100

def calc_energy_utilization(original, optimized):
    curtailed = np.sum(original - optimized[optimized < original])
    total_available = np.sum(original)
    return (1 - curtailed/total_available) * 100

peak_reduction = calc_peak_reduction(original_load, optimized_load)
ramp_reduction = calc_ramp_reduction(original_load, optimized_load)
energy_utilization = calc_energy_utilization(original_load, optimized_load)
avg_reward = np.mean(rewards)

kpis = {
    "Peak Reduction (%)": peak_reduction,
    "Ramp Reduction (%)": ramp_reduction,
    "Renewable Utilization (%)": energy_utilization,
    "Average Reward": avg_reward
}

print("\n=== RL Evaluation KPIs ===")
for k, v in kpis.items():
    print(f"{k}: {v:.2f}")

# Optional: save KPI table
df_kpis = pd.DataFrame([kpis])
df_kpis.to_csv(os.path.join(data_folder, "kpis_summary.csv"), index=False)
