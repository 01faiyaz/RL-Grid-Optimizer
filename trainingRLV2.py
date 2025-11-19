from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env.gridEnvV2 import GridEnvV2

#creating the second environment
env = GridEnvV2()

#making the model 
#attains the ppo model and parameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    tensorboard_log=None,  # disables tensorboard
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
)


# calling checkpoint callback to generate checkpoints by every 100k steps
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="./models/",
    name_prefix="ppo_grid_ckpt"
)

# 500k steps for training for more accuracy
model.learn(
    total_timesteps=500_000,
    callback=checkpoint_callback
)


model.save("models/ppo_grid_model_v2")
