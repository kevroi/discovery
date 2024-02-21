import os
import numpy as np
import torch
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, FullyObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor
import matplotlib.pyplot as plt

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
    x = x + 1e-8 # add epsilon to avoid division by zero
    x = x / torch.norm(x, dim=1) # unit vector
    return x


def cosine_similarity(phi, phi_goal):
    # assuming phi and phi_goal are unit vectors
    return torch.dot(phi, phi_goal)


def cosine_similarity_matrix(feature_activations):
    # assuming feature_activations is a tensor of unit vectors
    return torch.mm(feature_activations, feature_activations.T)

def plot_average_heatmap(agent_name, env_name, folder_path):
    # Initialize variables to store the sum of matrices and the count of matrices
    sum_matrix = None
    count = 0

    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith(agent_name) and filename.endswith(".npy"):
            matrix_data = np.load(os.path.join(folder_path, filename))
            # matrix_data = np.nan_to_num(matrix_data, nan=0.0) # replace NaN with 0
            if sum_matrix is None:
                sum_matrix = matrix_data
            else:
                sum_matrix += matrix_data
            count += 1
    average_matrix = sum_matrix / count

    plt.figure(figsize=(10, 8))
    plt.imshow(average_matrix, cmap='viridis', interpolation='nearest')
    plt.title(f'Average Heatmap of {agent_name} on {env_name}')
    plt.colorbar()
    plt.show()


def get_subgoal_index(config):
    """Along the optimal trajectory, this function returns the timestep of the subgoal.
    This is the index of the observation in the feature_activation matrix.
    """
    if config['env_name'] == 'MiniGrid-DoorKey-5x5-v0':
        subgoal_indices = [2, 6] # after pickuing up key, after opening door
    else:
        raise ValueError(f"Subgoal index not implemented for {config['env_name']}.")
    
    return subgoal_indices


def check_directory(directory):
    if os.path.exists(directory):
        pass
    else:
        os.makedirs(directory)