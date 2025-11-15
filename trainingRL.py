import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from env.gridEnv import GridEnv  # your environment

# ----------------------------------------
# 1. Create environment
# ----------------------------------------
env = GridEnv()
env = Monitor(env)

# ----------------------------------------
# 2. Create PPO model
# ----------------------------------------
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
)

# ----------------------------------------
# 3. Train
# ----------------------------------------
model.learn(total_timesteps=200_000)

# ----------------------------------------
# 4. Save the model
# ----------------------------------------
model.save("models/ppo_grid_model")

print("✔️ Training complete and model saved.")
