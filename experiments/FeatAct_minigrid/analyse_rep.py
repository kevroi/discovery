import numpy as np
import torch
import torchvision
import zipfile
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import *
import matplotlib.pyplot as plt
import wandb

def get_obs(env, see_obs=False):
    print(env.get_attr("spec")[0].id)
    if env.get_attr("spec")[0].id == 'MiniGrid-DoorKey-5x5-v0':
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

        # TODO: modify this to take snapshots of the environment at different points in time
        # if see_obs:
        # img = env.render()
        # plt.figure()
        # plt.imshow(np.concatenate([img], 1)) # shows the full environment
        # plt.savefig("../../plots/domains/DoorKey_5x5.pdf", dpi=300)
    
    if env.get_attr("spec")[0].id == 'MiniGrid-DoorKey-8x8-v0':
        obs_list = []
        obs = env.reset()
        action_seq = [2, 0, 3, 2, 0, 2, 2, 2, 1, 5, 2, 2, 1, 2, 2, 2]
        for a in action_seq:
            obs, _, _, _ = env.step([a])
            obs_list.append(obs)
    else:
        raise ValueError(f"Analysis not implemented for {env.get_attr('spec')[0].id}.")
    
    return obs_list


def get_feats(model, config, see_bad_obs=False):
    env = DummyVecEnv([lambda: make_env(config=config)])
    env.seed(0)
    if see_bad_obs:
        obs_list = get_bad_obs(env)
    else:
        obs_list = get_obs(env, see_obs=True)
    feature_activations = []

    with torch.no_grad():
        for obs in obs_list:
            phi = extract_feature(model, obs)
            feature_activations.append(phi)

    feature_activations = torch.cat(feature_activations, dim=0)
    cos_sim_matrix = cosine_similarity_matrix(feature_activations)

    if config['use_wandb']:
        images = wandb.Image(feature_activations, caption="Feature Activations")
        wandb.log({"Feat_act": images})
        for i, index in enumerate(get_subgoal_index(config)):
            phi_subgoal = feature_activations[index]
            for j, phi in enumerate(feature_activations):
                wandb.log({f"Cosine Similarity with phi_subgoal_{i}": cos_sim_matrix[index, j]})
        labels = [f"phi(o_{i})" for i in range(len(obs_list))]
        wandb.log({"CosSim": wandb.plots.HeatMap(labels, labels, cos_sim_matrix, show_text=False)})

    # save the cosine similarity matrix
    save_path  = f"experiments/FeatAct_minigrid/cos_sim_matrices"
    check_directory(save_path)
    np.save(save_path +f"/{config['learner']}_{config['env_name']}_{config['feat_dim']}feats_{str(config['run_num'])}.npy", cos_sim_matrix.numpy())

    return feature_activations

def see_feats(feature_activations):
    plt.figure()
    plt.imshow(feature_activations, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()

def save_feats(feature_activations, config):
    save_path  = f"experiments/FeatAct_minigrid/feat_acts/feature_activations"
    np.save(save_path +f"_{config['env_name']}.npy", feature_activations.numpy()) # TODO: should add a timestamp/hash to the filename
    with zipfile.ZipFile('array.zip', 'w') as zipf:
        zipf.write('array.npy')
