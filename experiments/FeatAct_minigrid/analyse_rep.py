import numpy as np
import torch
import torchvision
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import make_env
import matplotlib.pyplot as plt
import wandb

## HELPER FUNCTIONS ##
def pre_process_obs(obs, model):
    obs = np.transpose(obs, (0,3,1,2)) # bring colour channel to front
    return obs_as_tensor(obs, model.policy.device)

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
    max_feat_list = []
    feature_activations = []

    with torch.no_grad():
        for obs in obs_list:
            obs = pre_process_obs(obs, model)
            if model.__class__.__name__ == "DoubleDQN":
                x = model.policy.extract_features(obs, model.policy.q_net.features_extractor)
            elif model.__class__.__name__ == "PPO":
                x = model.policy.extract_features(obs)/255
            else:
                raise ValueError(f"Feature extractor for {model.__class__.__name__} not implemented.")
            max_feat_list.append(torch.argmax(x).item())
            x_ = x.reshape(1, -1)
            feature_activations.append(x_)

    feature_activations = torch.cat(feature_activations, dim=0)
    images = wandb.Image(feature_activations, caption="Feature Activations")
    wandb.log({"Feat_act": images})

    return feature_activations, max_feat_list