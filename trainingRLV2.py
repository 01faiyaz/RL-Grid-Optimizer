from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env.gridEnvV2 import GridEnvV2

# ---- Create Environment ----
env = GridEnvV2()

# ---- Model ----
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    tensorboard_log=None,  # disables tensorboard
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
)


# ---- Save checkpoints every 100k steps ----
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="./models/",
    name_prefix="ppo_grid_ckpt"
)

# ---- Train ----
model.learn(
    total_timesteps=500_000,
    callback=checkpoint_callback
)

# ---- Save final model ----
model.save("models/ppo_grid_model_v2")
print("\nTraining complete → model saved!")
