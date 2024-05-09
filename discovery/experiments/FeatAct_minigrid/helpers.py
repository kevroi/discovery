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
    if config['env_name'] == "FourRoomChainEnv":
        from environments.custom_minigrids import FourRoomChainEnv
        gym.register(id="FourRoomChainEnv", entry_point=FourRoomChainEnv)
        env = gym.make(config['env_name'], render_mode=config['render_mode'])
        env = FullyObsWrapper(env) # FullyObsWrapper runs faster locally, but uses ints instead of 256-bit RGB
        env = ImgObsWrapper(env)
    elif config['env_name'] == "TwoRoomEnv":
        from discovery.environments.custom_minigrids import TwoRoomEnv
        gym.register(id="TwoRoomEnv", entry_point=TwoRoomEnv)
        env = gym.make(config['env_name'], render_mode=config['render_mode'], random_hallway=config['random_hallway'])
        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
    elif config['env_name'] == "FourRoomEnv":
        from discovery.environments.custom_minigrids import FourRoomEnv
        gym.register(id="FourRoomEnv", entry_point=FourRoomEnv)
        env = gym.make(config['env_name'], render_mode=config['render_mode'])
        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
    elif config['env_name'] == "discovery/Climbing-v0":
        from discovery.environments.climbing import ClimbingEnv
        gym.register(id="discovery/Climbing-v0", entry_point="discovery.environments.climbing:ClimbingEnv")
        env = gym.make(config['env_name'])
    else:
        env = gym.make(config['env_name'], render_mode=config['render_mode'])
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
    env = Monitor(env)
    return env


## HELPER FUNCTIONS ##
def pre_process_obs(obs, model):
    if obs.ndim == 3:
        obs = np.expand_dims(obs, axis=0) # add batch dimension if its just one observation
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

def plot_average_heatmap(agent_name, env_name, folder_path, plot_sg_cossim=False, feat_dim=8, activation=None):
    # Initialize variables to store the sum of matrices and the count of matrices
    sum_matrix = None
    std_error_matrix = None
    count = 0

    if activation is not None:
        folder_path = folder_path + f"_{activation}"

    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith(f"{agent_name}_{env_name}_{feat_dim}") and filename.endswith(".npy"):
            matrix_data = np.load(os.path.join(folder_path, filename))
            if sum_matrix is None:
                sum_matrix = matrix_data
            else:
                sum_matrix += matrix_data
            count += 1
    average_matrix = sum_matrix / count
    plt.figure(figsize=(10, 8))
    plt.imshow(average_matrix, cmap='gist_heat', interpolation='nearest')
    plt.title(f'Average Heatmap of {agent_name} on {env_name} with {feat_dim} features')
    plt.clim(0.5,1.0)
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=18)
    # plt.show()
    plt.savefig(f"plots/feature_activations/minigrid_doorkey_5x5/cossim_{agent_name}_{env_name}_{feat_dim}feats_{activation}.pdf")

    if plot_sg_cossim:
        # iterate over files again to get the std error
        for filename in os.listdir(folder_path):
            if filename.startswith(agent_name) and filename.endswith(".npy"):
                matrix_data = np.load(os.path.join(folder_path, filename))
                matrix_data = np.nan_to_num(matrix_data, nan=0.0)
                np.fill_diagonal(matrix_data, 1.0)
                if std_error_matrix is None:
                    std_error_matrix = (matrix_data - average_matrix)**2
                else:
                    std_error_matrix += (matrix_data - average_matrix)**2
        std_error_matrix = np.sqrt(std_error_matrix / count)
        # Plot the cosine similarity of the subgoals
        subgoal_indices = get_subgoal_index({'env_name': env_name})
        for i, index in enumerate(subgoal_indices):
            plt.figure()
            plt.plot(average_matrix[index, :])
            plt.fill_between(range(average_matrix.shape[0]),
                             np.maximum(average_matrix[index, :] - std_error_matrix[index, :], np.zeros(average_matrix.shape[0])),
                             np.minimum(average_matrix[index, :] + std_error_matrix[index, :], np.ones(average_matrix.shape[0])),
                             alpha=0.3)
            plt.ylim(0.5, 1)
            plt.title(f'Cosine Similarity with subgoal {i}')
            plt.xlabel('Timestep')
            plt.ylabel('Cosine Similarity')
            plt.show()

def plot_identity_heatmap():
    identity_matrix = np.identity(11)
    plt.figure(figsize=(10, 8))
    plt.imshow(identity_matrix, cmap='gist_heat', interpolation='nearest')
    plt.title(f'Identity Heatmap')
    plt.clim(0.5,1.0)
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=18)
    # plt.show()
    plt.savefig(f"plots/feature_activations/minigrid_doorkey_5x5/tabular.pdf")


def plot_sg_cossim(agent_name, env_name, folder_path, feat_dims=[120]):
    subgoal_indices = get_subgoal_index({'env_name': env_name})
    for i, index in enumerate(subgoal_indices):
        plt.figure()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        plt.ylim(0.0, 1)
        # plt.title(f'Cosine Similarity with subgoal {i}')
        plt.xlabel('Timestep')
        plt.ylabel('Cosine Similarity')
        for feat_dim in feat_dims:
            # Initialize variables to store the sum of matrices and the count of matrices
            sum_matrix = None
            std_error_matrix = None
            count = 0

            #Iterate over the files in the folder
            for filename in os.listdir(folder_path):
                if filename.startswith(f"{agent_name}_{env_name}_{feat_dim}") and filename.endswith(".npy"):
                    matrix_data = np.load(os.path.join(folder_path, filename))
                    if sum_matrix is None:
                        sum_matrix = matrix_data
                    else:
                        sum_matrix += matrix_data
                    count += 1
            average_matrix = sum_matrix / count

            # Iterate over the files in the folder
            for filename in os.listdir(folder_path):
                if filename.startswith(f"{agent_name}_{env_name}_{feat_dim}") and filename.endswith(".npy"):
                    matrix_data = np.load(os.path.join(folder_path, filename))
                    if std_error_matrix is None:
                        std_error_matrix = (matrix_data - average_matrix)**2
                    else:
                        std_error_matrix += (matrix_data - average_matrix)**2
                    count += 1
            std_error_matrix = np.sqrt(std_error_matrix / count)

            plt.plot(average_matrix[index, :], label=f"{feat_dim} features")
            plt.fill_between(range(average_matrix.shape[0]),
                                np.maximum(average_matrix[index, :] - std_error_matrix[index, :], np.zeros(average_matrix.shape[0])),
                                np.minimum(average_matrix[index, :] + std_error_matrix[index, :], np.ones(average_matrix.shape[0])),
                                alpha=0.3)
        plt.show()
        plt.close()



def plot_sg_cossim_diff_act(agent_name, env_name, folder_paths, feat_dims=[12]):
    subgoal_indices = get_subgoal_index({'env_name': env_name})
    colors = {"experiments/FeatAct_minigrid/cos_sim_matrices_relu": "#0077BB",
          "experiments/FeatAct_minigrid/cos_sim_matrices_fta": "#EE3377",
          "experiments/FeatAct_minigrid/cos_sim_matrices_crelu": "#EE7733",
          "512": "#009988"}
    labels = {"experiments/FeatAct_minigrid/cos_sim_matrices_relu": "ReLU",
          "experiments/FeatAct_minigrid/cos_sim_matrices_fta": "FTA",
          "experiments/FeatAct_minigrid/cos_sim_matrices_crelu": "CReLU",
          }
    for i, index in enumerate(subgoal_indices):
        plt.figure()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        plt.ylim(0.0, 1)
        # plt.title(f'Cosine Similarity with subgoal {i}')
        plt.xlabel('Timestep')
        plt.ylabel('Cosine Similarity')
        for folder_path in folder_paths:
            for feat_dim in feat_dims:
                # Initialize variables to store the sum of matrices and the count of matrices
                sum_matrix = None
                std_error_matrix = None
                count = 0

                #Iterate over the files in the folder
                for filename in os.listdir(folder_path):
                    if filename.startswith(f"{agent_name}_{env_name}_{feat_dim}") and filename.endswith(".npy"):
                        matrix_data = np.load(os.path.join(folder_path, filename))
                        if sum_matrix is None:
                            sum_matrix = matrix_data
                        else:
                            sum_matrix += matrix_data
                        count += 1
                average_matrix = sum_matrix / count

                # Iterate over the files in the folder
                for filename in os.listdir(folder_path):
                    if filename.startswith(f"{agent_name}_{env_name}_{feat_dim}") and filename.endswith(".npy"):
                        matrix_data = np.load(os.path.join(folder_path, filename))
                        if std_error_matrix is None:
                            std_error_matrix = (matrix_data - average_matrix)**2
                        else:
                            std_error_matrix += (matrix_data - average_matrix)**2
                        count += 1
                std_error_matrix = np.sqrt(std_error_matrix / count)
                std_error_matrix = 1.96 * std_error_matrix / np.sqrt(count)

                plt.plot(average_matrix[index, :], label=labels[folder_path], color=colors[folder_path])
                plt.fill_between(range(average_matrix.shape[0]),
                                    np.maximum(average_matrix[index, :] - std_error_matrix[index, :], np.zeros(average_matrix.shape[0])),
                                    np.minimum(average_matrix[index, :] + std_error_matrix[index, :], np.ones(average_matrix.shape[0])),
                                    alpha=0.3, color=colors[folder_path])
        plt.legend()
        # plt.show()
        plt.savefig(f"plots/cos_sim_line/minigrid_doorkey_5x5/cossimline_{agent_name}_{env_name}_sg{i}_acts.pdf")
        plt.close()

def plot_sg_cossim_diff_feats(agent_name, env_name, folder_path, feat_dims=[8, 32]):
    # # BLUE Gradient
    # colors = {"2": "#79abe1",
    #       "32": "#0077BB",
    #       "256": "#00356c",
    #       "512": "#009988"}

    # # Orange Gradient
    # colors = {"4": "#fec44f",
    #       "32": "#EE7733",
    #       "256": "#993404"}
    
    # # Pink Gradient
    colors = {
          "40": "#EE3377",
          "160": "#cd0bbc",
          "640": "#661100"}

    subgoal_indices = get_subgoal_index({'env_name': env_name})
    for i, index in enumerate(subgoal_indices):
        plt.figure()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        plt.ylim(0.0, 1)
        # plt.title(f'Cosine Similarity with subgoal {i}')
        plt.xlabel('Timestep')
        plt.ylabel('Cosine Similarity')
        for feat_dim in feat_dims:
            # Initialize variables to store the sum of matrices and the count of matrices
            sum_matrix = None
            std_error_matrix = None
            count = 0

            #Iterate over the files in the folder
            for filename in os.listdir(folder_path):
                if filename.startswith(f"{agent_name}_{env_name}_{feat_dim}") and filename.endswith(".npy"):
                    matrix_data = np.load(os.path.join(folder_path, filename))
                    if sum_matrix is None:
                        sum_matrix = matrix_data
                    else:
                        sum_matrix += matrix_data
                    count += 1
            average_matrix = sum_matrix / count

            # Iterate over the files in the folder
            for filename in os.listdir(folder_path):
                if filename.startswith(f"{agent_name}_{env_name}_{feat_dim}") and filename.endswith(".npy"):
                    matrix_data = np.load(os.path.join(folder_path, filename))
                    if std_error_matrix is None:
                        std_error_matrix = (matrix_data - average_matrix)**2
                    else:
                        std_error_matrix += (matrix_data - average_matrix)**2
                    count += 1
            std_error_matrix = np.sqrt(std_error_matrix / count)

            # calciualte 95% confidence interval
            std_error_matrix = 1.96 * std_error_matrix / np.sqrt(count)

            plt.plot(average_matrix[index, :], label=f"{feat_dim} features", color = colors[str(feat_dim)])
            plt.fill_between(range(average_matrix.shape[0]),
                                np.maximum(average_matrix[index, :] - std_error_matrix[index, :], np.zeros(average_matrix.shape[0])),
                                np.minimum(average_matrix[index, :] + std_error_matrix[index, :], np.ones(average_matrix.shape[0])),
                                alpha=0.3, color=colors[str(feat_dim)])
        plt.legend()
        # plt.show()
        plt.savefig(f"plots/cos_sim_line/minigrid_doorkey_5x5/cossimline_{agent_name}_{env_name}_sg{i}_fta.pdf")
        plt.close()




def get_subgoal_index(config):
    """Along the optimal trajectory, this function returns the timestep of the subgoal.
    This is the index of the observation in the feature_activation matrix.
    """
    if config['env_name'] == 'MiniGrid-DoorKey-5x5-v0':
        subgoal_indices = [2, 6] # after pickuing up key, after opening door
    elif config['env_name'] == 'MiniGrid-DoorKey-8x8-v0':
        subgoal_indices = [3, 10] # after pickuing up key, after opening door
    else:
        raise ValueError(f"Subgoal index not implemented for {config['env_name']}.")
    
    return subgoal_indices


def check_directory(directory):
    if os.path.exists(directory):
        pass
    else:
        os.makedirs(directory)

if __name__=="__main__":
    # plot_average_heatmap("PPO", "MiniGrid-DoorKey-5x5-v0", "experiments/FeatAct_minigrid/cos_sim_matrices",
                        #  feat_dim=64, plot_sg_cossim=False, activation="crelu")
    # plot_sg_cossim("PPO", "MiniGrid-DoorKey-5x5-v0", "experiments/FeatAct_minigrid/cos_sim_matrices", feat_dims=[8, 32])
    # plot_sg_cossim_diff_act("PPO", "MiniGrid-DoorKey-5x5-v0", ["experiments/FeatAct_minigrid/cos_sim_matrices_relu", "experiments/FeatAct_minigrid/cos_sim_matrices_fta", "experiments/FeatAct_minigrid/cos_sim_matrices_crelu"], feat_dims=[40])
    plot_sg_cossim_diff_feats("PPO", "MiniGrid-DoorKey-5x5-v0", "experiments/FeatAct_minigrid/cos_sim_matrices_fta", feat_dims=[40, 160, 640])
    # plot_identity_heatmap()