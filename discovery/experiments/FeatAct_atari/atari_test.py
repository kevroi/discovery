from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
import gymnasium as gym
import wandb
from discovery.utils.save_callback import SnapshotCallback

print("modules loaded")

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
vec_env = make_atari_env("ALE/Seaquest-v5", n_envs=1, seed=0)
# Frame-stacking with 4 frames
# vec_env = VecFrameStack(vec_env, n_stack=4)
print("made env")

log_directory = "discovery/experiments/FeatAct_atari/models/test"
snapshot_callback = SnapshotCallback(check_freq=1_000, log_dir=log_directory, verbose=1)

model = PPO("CnnPolicy", vec_env, verbose=1)
print("made model")
model.learn(total_timesteps=10_000, callback=snapshot_callback)
print("model trained")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=False)
#     obs, rewards, dones, info = vec_env.step(action)
#     print(rewards)
#     # vec_env.render("human")
