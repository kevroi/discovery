import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
from gymnasium.experimental.vector import VectorRewardWrapper
from experiments.FeatAct_minigrid.utils import extract_feature


class StoppingBonusWrapper(VecEnvWrapper):
    def __init__(self, venv, agent, target_feature, reward_bonus):
        VecEnvWrapper.__init__(self, venv)
        self.agent = agent
        self.target_feature = (
            target_feature  # load this from experiments training CNN alongside PPO/DDQN
        )
        self.reward_bonus = reward_bonus
        self.epsilon_closeness = 0.1

        # # Modify observation space to include the feature vector
        # self.observation_space = spaces.Box(
        #     low=float('-inf'),
        #     high=float('inf'),
        #     shape=(venv.observation_space.shape[0] + len(target_feature),),
        #     dtype=np.float32
        # )

    # def reset(self):
    #     obs = self.venv.reset()
    #     # Compute the feature vector from the CNN model
    #     feature_vector = extract_feature(self.agent, obs)
    #     # self.current_obs = np.concatenate([obs, feature_vector])
    #     return self.current_obs

    def step(self, action):
        obs, reward, done, info = self.venv.step(action)
        # Compute the feature vector from the CNN model
        feature_vector = extract_feature(self.agent, obs)

        # Compute reward bonus based on similarity to target feature
        # similarity = np.dot(feature_vector, self.target_feature)
        l_1_dist = np.linalg.norm(feature_vector - self.target_feature, ord=1)
        if l_1_dist < self.epsilon_closeness:
            reward += self.reward_bonus

        return self.current_obs, reward, done, info
