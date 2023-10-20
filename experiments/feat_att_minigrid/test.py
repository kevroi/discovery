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
    "env_name": "MiniGrid-Empty-16x16-v0",
}

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

# load agent
model = PPO.load("models/lmqa7hgs/model.zip")
print(type(model.policy))
print(type(model.policy.features_extractor))
print(model.policy.share_features_extractor)



# recreate vector environment - observations from vectorEnvs are 4D tensors. The CNN expects this
def make_env():
    env = gym.make(config["env_name"], render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

# model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
obs = env.reset()
before_img = env.render()
obs = np.transpose(obs, (0,3,1,2)) # bring colour channel to front
 
# print(len(obs))
# obs = obs.permute(2,0,1)
# print(init_obs_tensor.shape)
# print(type(model.policy.features_extractor))
# obs, _, _, _ = env.step([0])
# model.policy.set_training_mode(False)
with torch.no_grad():
    obs = obs_as_tensor(obs, model.policy.device)
    print(obs.shape) 
    x = model.policy.extract_features(obs) # TODO figure out what shape my observation passed in should be, is it a batch of obs? CNN policy doesnt seem to have it.
print(x.shape)
# action = env.actions.forward
# obs, reward, terminated, info, done = env.step(action)
# obs, reward, terminated, info, done = env.step(env.actions.right)
# for i in range(7):
#     obs, reward, terminated, info, done = env.step(action)
# obs, reward, terminated, info, done = env.step(env.actions.left)
# after_img = env.render()

# print(obs.shape)
# # print(obs)

# plt.imshow(np.concatenate([before_img, after_img], 1)) # shows the full environment
# plt.show()

plt.imshow(x.reshape(1, -1), cmap='viridis', aspect='auto', interpolation='none')
plt.colorbar()
plt.xlabel('Value Index')
plt.title('Vertical Heatmap of 1D Tensor')
plt.show()