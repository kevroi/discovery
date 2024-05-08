from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
import gymnasium as gym
import wandb
print("modules loaded")

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
vec_env = make_atari_env("ALE/MsPacman-v5", n_envs=4, seed=0)
# Frame-stacking with 4 frames
vec_env = VecFrameStack(vec_env, n_stack=4)
print("made env")

model = PPO("CnnPolicy", vec_env, verbose=1)
print("made model")
model.learn(total_timesteps=25_000)
print("model trained")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    print(rewards)
    # vec_env.render("human")