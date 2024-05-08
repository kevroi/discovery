import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn.functional as F

class ClimbingFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        height = observation_space["agent_loc"].n
        super(ClimbingFeatureExtractor, self).__init__(observation_space, features_dim=height+2)

    def forward(self, observations):
        features = torch.cat((observations["agent_loc"], observations["at_anchor"]), dim=-1)
        return features