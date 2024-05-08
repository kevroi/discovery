import sys
import numpy as np
import pickle
import copy
from tqdm import tqdm

import utils.rlglue as rlglue
from environments.environment import ToyEnvironment, GridEnvironment, I_MazeEnvironment
from agents.TD_lambda import TDLambdaAgent
from eigenoptions.options import Options

# env = GridEnvironment()
env = ToyEnvironment('2room_lava')
max_row, max_col = env.get_grid_dimension() 
agent = TDLambdaAgent(max_row=max_row, max_col=max_col)
agent.set_alpha(0.1)
agent.set_discount(0.9)
agent.set_lmbda(0.9)
# agent.set_subgoals([(2,5), (6, 8), (9,5), (5,1)]) # 4rooms
agent.set_subgoals([(2,6)]) # 2rooms
glue = rlglue.RLGlue(env, agent)

num_runs = 1
num_episodes = 1
cum_reward = np.zeros(num_episodes)

for run in tqdm(range(num_runs)):
    for ep in range(num_episodes):
        # run episode
        glue.episode(100)
        learned_V = agent.get_V()
        agent.set_V(learned_V)
        glue.episode(100)
        cum_reward[ep] += glue.get_total_reward()
        glue.cleanup()
cum_reward /= float(num_runs)

np.save(f'experiments/feature_attainment/data_files/{env.name}_average_return', cum_reward)

