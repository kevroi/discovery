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
from helpers import make_env
from stable_baselines3.common.utils import obs_as_tensor
from environments.custom_minigrids import TwoRoomEnv, FourRoomEnv, FourRoomChainEnv
import pygame

config = {
    "env_name": "TwoRoomEnv",
    # "env_name": "FourRoomEnv",
    "random_hallway": False,
    "render_mode": "human",
}

env = make_env(config)
manual_control = ManualControl(env)
manual_control.start()