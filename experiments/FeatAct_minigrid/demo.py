import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from minigrid.manual_control import ManualControl
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from utils.cnn import MinigridFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor
from experiments.feat_att_minigrid.n_room_env import TwoRoomEnv, FourRoomEnv, FourRoomChainEnv
import pygame


# Helpers
def make_env():
    env = gym.make(config["env_name"], render_mode="human")
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env

def pre_process_obs(obs, model):
    obs = np.transpose(obs, (0,3,1,2)) # bring colour channel to front
    return obs_as_tensor(obs, model.policy.device)

def show_feats():
    x = np.arange(0, 1, 0.01)
    y = np.sin(2 * np.pi * x)
    plt.plot(x, y)
    plt.show()

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 2e5,
    # "env_name": "MiniGrid-Empty-16x16-v0",
    "env_name": "MiniGrid-Empty-5x5-v0",
    "feat_dim":32,
}

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=config["feat_dim"]),
)

model = PPO.load("models/fmayv5p9/model.zip")
env = gym.make(config["env_name"], render_mode="human", tile_size=200)
# env = FourRoomChainEnv(render_mode="human", random_goal=True)
manual_control = ManualControl(env)
manual_control.start()

for event in pygame.event.get():
    if event.type == pygame.KEYDOWN:
        show_feats()