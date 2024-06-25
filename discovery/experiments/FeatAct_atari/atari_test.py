import time
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
import gymnasium as gym
import wandb
from discovery.utils.save_callback import SnapshotCallback
from discovery.utils import filesys


print("modules loaded")

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
vec_env = make_atari_env("ALE/Seaquest-v5", n_envs=1, seed=0)
# Frame-stacking with 4 frames
vec_env = VecFrameStack(vec_env, n_stack=4)
print("made env")

log_directory = "discovery/experiments/FeatAct_atari/models"
snapshot_callback = SnapshotCallback(check_freq=1_000, log_dir=log_directory, verbose=1)

# model = PPO("CnnPolicy", vec_env, verbose=1)
# print("made model")
# model.learn(total_timesteps=10_000, callback=snapshot_callback)
# print("model trained")


orig_full_path = "/Users/kevinroice/Documents/research/discovery/discovery/experiments/FeatAct_atari/models/test/model_snapshot_9000_steps.zip"
seaquest_paths = [
    "discovery/experiments/FeatAct_atari/models/seaquest_cnn/Seaquest-v5_mpqgvvr1.zip",  # 0, sticks to lower half, shoots fish
    "discovery/experiments/FeatAct_atari/models/PPO_ALE/Seaquest-v5_4oje7yx4.zip",  # 1, sticks to the bottom, shoots fish
    "discovery/experiments/FeatAct_atari/models/PPO_ALE/Seaquest-v5_8r01d3y7.zip",  # 2, sticks to the lower half, shoots fish
    "discovery/experiments/FeatAct_atari/models/PPO_ALE/Seaquest-v5_ekqjr6df.zip",  # 3, same, sometimes moves upwards to get air
    "discovery/experiments/FeatAct_atari/models/PPO_ALE/Seaquest-v5_gciysixg.zip",  # 4,
    "discovery/experiments/FeatAct_atari/models/PPO_ALE/Seaquest-v5_gux8ixql.zip",  # 5, same
    "discovery/experiments/FeatAct_atari/models/PPO_ALE/Seaquest-v5_itafwycc.zip",
    "discovery/experiments/FeatAct_atari/models/PPO_ALE/Seaquest-v5_o8ec1u1z.zip",
    "discovery/experiments/FeatAct_atari/models/PPO_ALE/Seaquest-v5_oqy9kqe9.zip",
    "discovery/experiments/FeatAct_atari/models/PPO_ALE/Seaquest-v5_rmfyq9p4.zip",
    "discovery/experiments/FeatAct_atari/models/PPO_ALE/Seaquest-v5_w53h6u4k.zip",
]
full_path = filesys.make_abs_path_in_root(seaquest_paths[6])
model2 = PPO.load(full_path)

obs = vec_env.reset()
while True:
    action, _states = model2.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    print(rewards)
    vec_env.render("human")
    time.sleep(0.1)
