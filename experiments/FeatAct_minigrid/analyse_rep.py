import numpy as np
import torch
import torchvision
import zipfile
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import make_env, extract_feature, cosine_similarity, get_subgoal_index
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
    
    else:
        raise ValueError(f"Analysis not implemented for {env.get_attr('spec')[0].id}.")
    
    return obs_list


def get_feats(model, config):
    env = DummyVecEnv([lambda: make_env(config=config)])
    env.seed(0)
    obs_list = get_obs(env, see_obs=True)
    feature_activations = []

    with torch.no_grad():
        for obs in obs_list:
            phi = extract_feature(model, obs)
            feature_activations.append(phi)

    feature_activations = torch.cat(feature_activations, dim=0)

    if config['use_wandb']:
        images = wandb.Image(feature_activations, caption="Feature Activations")
        wandb.log({"Feat_act": images})
        phi_subgoal = feature_activations[get_subgoal_index(config)]
        for phi in feature_activations:
            wandb.log({"Cosine Similarity with phi_subgoal": cosine_similarity(phi, phi_subgoal)})

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
