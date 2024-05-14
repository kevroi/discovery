import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from stable_baselines3 import PPO
from discovery.utils.feat_extractors import MinigridFeaturesExtractor, NatureCNN
from stable_baselines3.common.utils import obs_as_tensor
