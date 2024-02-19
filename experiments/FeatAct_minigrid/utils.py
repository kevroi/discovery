import numpy as np
import torch
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, FullyObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor

# HELPER FUNCTIONS ##
def make_env(config):
    env = gym.make(config['env_name'], render_mode='rgb_array')
    env = RGBImgObsWrapper(env) # FullyObsWrapper runs faster locally, but uses ints instead of 256-bit RGB
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env


## HELPER FUNCTIONS ##
def pre_process_obs(obs, model):
    obs = np.transpose(obs, (0,3,1,2)) # bring colour channel to front
    return obs_as_tensor(obs, model.policy.device)


def extract_feature(agent, obs):
    # assuming this function is used within a torch.no_grad() context
    obs = pre_process_obs(obs, agent)
    if agent.__class__.__name__ == "DoubleDQN":
        x = agent.policy.extract_features(obs, agent.policy.q_net.features_extractor)
    elif agent.__class__.__name__ == "PPO":
        x = agent.policy.extract_features(obs)/255
    else:
        raise ValueError(f"Feature extraction for {agent.__class__.__name__} not implemented.")
    x = x.reshape(1, -1)
    x = x.cpu()
    x = x / torch.norm(x, dim=1) # unit vector
    x = x.detach().numpy()
    return x


def cosine_similarity(phi, phi_goal):
    # assuming phi and phi_goal are unit vectors
    return np.dot(phi, phi_goal)


def get_subgoal_index(config):
    """Along the optimal trajectory, this function returns the timestep of the subgoal.
    This is the index of the observation in the feature_activation matrix.
    """
    if config['env_name'] == 'MiniGrid-DoorKey-5x5-v0':
        subgoal_index = 5
    else:
        raise ValueError(f"Subgoal index not implemented for {config['env_name']}.")
    
    return subgoal_index