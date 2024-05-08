import gymnasium as gym
from gymnasium import spaces
from discovery.environments import climbing
from discovery.utils.climbing_feats import ClimbingFeatureExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# height = 8
# num_envs = 4
# env_id = "discovery/Climbing-v0"
# env = gym.make("discovery/Climbing-v0")
# observation, info = env.reset(seed=42)
# print("observation: ", observation)

# policy_kwargs = dict(
#                     features_extractor_class=ClimbingFeatureExtractor,
#                     # features_extractor_kwargs=dict(observation_space=env.observation_space),
#                     )
# observation_space = spaces.Dict({
#     "agent_loc": spaces.Discrete(8),  # Assuming 8 possible agent locations
#     "at_anchor": spaces.Discrete(2)   # Binary indicator for anchor
# })
# agent = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# agent.learn(total_timesteps=10000)

## This shows the feature extractor works
# feature_extractor = ClimbingFeatureExtractor(observation_space)
# features = feature_extractor(observation)
# print(observation, features)

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv

# # Define the observation space as a dictionary
# observation_space = spaces.Dict({
#     "agent_loc": spaces.Discrete(8),  # Assuming 8 possible agent locations
#     "at_anchor": spaces.Discrete(2)   # Binary indicator for anchor
# })

# Create a single environment
env_id = "discovery/Climbing-v0"
env = gym.make(env_id)
n_envs = 1

# Wrap the environment in a VecEnv
env = DummyVecEnv([lambda: gym.make(env_id)]*n_envs)

# Create the feature extractor
# feature_extractor = ClimbingFeatureExtractor(observation_space)

policy_kwargs = dict(
                    features_extractor_class=ClimbingFeatureExtractor,
                    features_extractor_kwargs=dict(),
                    )

# Create the PPO agent
agent = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# x = agent.policy.extract_features(observation)
# Train the agent
agent.learn(total_timesteps=10000)

