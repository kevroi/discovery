import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from cnn import MinigridFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor


config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 2e5,
    # "env_name": "MiniGrid-Empty-16x16-v0",
    "env_name": "MiniGrid-Empty-5x5-v0",
    "feat_dim":32,
}

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=config["feat_dim"]),
)

# load agent
# model = PPO.load("models/lmqa7hgs/model.zip") # 16by16 empty
# model = PPO.load("models/ymxb53rz/model.zip") # 16by16 empty run 2
# model = PPO.load("models/41p5gtap/model.zip") # 5by5 empty feat vec = 128
model = PPO.load("models/fmayv5p9/model.zip")


print(type(model.policy))
print(type(model.policy.features_extractor))
print(model.policy.share_features_extractor)


# recreate vector environment - observations from vectorEnvs are 4D tensors. The CNN expects this
def make_env():
    env = gym.make(config["env_name"], render_mode="rgb_array")
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])
# model = PPO(config["policy_type"], env, policy_kwargs=policy_kwargs,
#             verbose=1, tensorboard_log=f"runs/test")

def pre_process_obs(obs, model):
    obs = np.transpose(obs, (0,3,1,2)) # bring colour channel to front
    return obs_as_tensor(obs, model.policy.device)

obs = env.reset()

# print(obs.shape)
# # print(obs)

# plt.imshow(np.concatenate([before_img, after_img], 1)) # shows the full environment
# plt.show()

def get_all_possible_obs(env, see_obs=False):
    obs = env.reset()
    # obs_list = [obs]
    obs_list = []
    for i in range(3):
        for j in range(3):
            for k in range(4):
                obs, _, _, _ = env.step([1])
                img = env.render()
                obs_list.append(obs)
                if see_obs:
                    plt.imshow(np.concatenate([img], 1)) # shows the full environment
                    plt.show()
            obs, _, _, _ = env.step([2])
            # obs_list.append(obs)
        env.reset()
        obs, _, _, _ = env.step([1])
        for down in range(i+1): # go to next row
            obs, _, _, _ = env.step([2])
        obs, _, _, _ = env.step([0])
    return obs_list

# plt.imshow(x.reshape(1, -1), cmap='viridis', aspect='auto', interpolation='none')
# plt.colorbar()
# plt.xlabel('Value Index')
# plt.title('Vertical Heatmap of 1D Tensor')
# plt.show()

obs_list = get_all_possible_obs(env, see_obs=False)
max_feat_list = []
feature_activations = []
# get_all_possible_obs(env)
# print(len(os))

with torch.no_grad():
    for obs in obs_list:
        obs = pre_process_obs(obs, model)
        x = model.policy.extract_features(obs)
        max_feat_list.append(torch.argmax(x).item())
        feature_activations.append(x.reshape(1, -1))

feature_activations = torch.cat(feature_activations, dim=0)

# print(feature_activations.shape)

# Show which features were activated the most across the obs space
# plt.hist(max_feat_list, bins=128)
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram of Feature indices')
# plt.show()

# Show which feature was activated for each obs
plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
plt.imshow(feature_activations, cmap='viridis', aspect='auto')
# y_tick_positions = [i * 56 for i in range(785 // 56)] # 16 by 16
y_tick_positions = [i * 12 for i in range(36 // 12)]
plt.yticks(y_tick_positions)
plt.colorbar()
plt.xlabel('Feature Index')
plt.ylabel('Observation Index')
plt.title('Feature Heatmap')

plt.show()
