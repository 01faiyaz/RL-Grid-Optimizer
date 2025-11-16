import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.gridEnv import GridEnv
from env.gridEnvV2 import GridEnvV2


# Create models folder
os.makedirs("models", exist_ok=True)

# Create environment
env = GridEnv(battery_power_kw=100)  # make battery more impactful
vec_env = make_vec_env(lambda: env, n_envs=1)

# Initialize PPO agent
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=0,        # 1 = basic logs, 0 = silent
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    clip_range=0.2,
)

# Train PPO agent
total_timesteps = 500_000  # increased for better learning
model.learn(total_timesteps=total_timesteps)

# Save trained model
model.save("models/ppo_grid_model")
print("✅ Model saved as models/ppo_grid_model.zip")
