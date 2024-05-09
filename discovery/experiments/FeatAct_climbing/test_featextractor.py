import gymnasium as gym
from gymnasium import spaces
from discovery.environments import climbing
from discovery.utils.feat_extractors import ClimbingFeatureExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


# Create a single environment
env_id = "discovery/Climbing-v0"
env = gym.make(env_id)
n_envs = 1

# Wrap the environment in a VecEnv
env = DummyVecEnv([lambda: gym.make(env_id)]*n_envs)

policy_kwargs = dict(
                    features_extractor_class=ClimbingFeatureExtractor,
                    features_extractor_kwargs=dict(),
                    )

# Create the PPO agent
agent = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# x = agent.policy.extract_features(observation)
# Train the agent
agent.learn(total_timesteps=10000)

