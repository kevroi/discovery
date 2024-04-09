import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from stable_baselines3 import PPO
from agents.ddqn import DoubleDQN
from cnn import MinigridFeaturesExtractor, NatureCNN
from stable_baselines3.common.utils import obs_as_tensor
from activations import CReLU, FTA

import random
import os
def make_env():
    # random.seed(0)
    # np.random.seed(0)
    # os.environ['PYTHONHASHSEED'] = str(0)
    # gym.utils.seeding.np_random(0)
    env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env

def pre_process_obs(obs, model):
    obs = np.transpose(obs, (0,3,1,2)) # bring colour channel to front
    return obs_as_tensor(obs, model.policy.device)

def get_obs(env, see_obs=False):
    # # Vector Action Encoding:
    # 0 = left
    # 1 = right
    # 2 = forward
    # 3 = pickup
    # 5 = activate an object (open door, press button)

    obs_list = []
    obs = env.reset() # initial observation
    obs_list.append(obs)
    
    # # move to the hallway
    obs, _, _, _ = env.step([1])
    obs_list.append(obs)# before picking up key
    obs, _, _, _ = env.step([3])
    obs_list.append(obs)# after picking up key
    obs, _, _, _ = env.step([2])
    obs_list.append(obs)
    obs, _, _, _ = env.step([2])
    obs_list.append(obs)
    obs, _, _, _ = env.step([1])
    obs_list.append(obs)# before opening door
    obs, _, _, _ = env.step([5])
    obs_list.append(obs)# after opening door
    obs, _, _, _ = env.step([2])
    obs_list.append(obs)
    obs, _, _, _ = env.step([2])
    obs_list.append(obs)
    obs, _, _, _ = env.step([1])
    obs_list.append(obs)
    obs, _, _, _ = env.step([2])
    obs_list.append(obs)# before reaching goal
    

    
    if see_obs:
        img = env.render()
        plt.figure()
        plt.imshow(np.concatenate([img], 1)) # shows the full environment
        # plt.savefig("../../plots/domains/DoorKey_5x5.pdf", dpi=300)
        plt.show()
    
    return obs_list


env = DummyVecEnv([make_env])
env.seed(0)
obs_list = get_obs(env, see_obs=False)
obs = obs_list[0]

policy_kwargs = dict(
                    features_extractor_class=NatureCNN,
                    features_extractor_kwargs=dict(features_dim=8,
                                                    # last_layer_activation=CReLU()),
                                                   last_layer_activation=FTA()),
                    )

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs,
            verbose=1)
obs = pre_process_obs(obs, model)
x = model.policy.extract_features(obs)
print(x.shape)