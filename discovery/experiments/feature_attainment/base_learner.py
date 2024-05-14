import sys
import numpy as np
import pickle
import copy
from tqdm import tqdm

import utils.rlglue as rlglue
from environments.environment import ToyEnvironment, GridEnvironment, I_MazeEnvironment
from agents.TD_lambda import TDLambdaAgent
from agents.Sarsa_lmbda import SarsaLmbdaAgent
from eigenoptions.options import Options


def see_action_values(Q):
    action_set = ["→", "←", "↓", "↑"]
    print(np.round(np.max(Q, axis=2), decimals=1))
    actions = np.argmax(Q, axis=2)
    opt_acts = np.zeros(actions.shape, dtype=str)
    for i in range(actions.shape[0]):
        for j in range(actions.shape[1]):
            opt_acts[i, j] = action_set[actions[i, j]]
    print(opt_acts)


# env = GridEnvironment()
env = ToyEnvironment("2room_lava")
max_row, max_col = env.get_grid_dimension()

# Figure 2 Agent
agent = SarsaLmbdaAgent(max_row=max_row, max_col=max_col)
agent.set_alpha(0.1)
agent.set_discount(0.9)
agent.set_epsilon(0.1)
agent.set_lmbda(0.9)
glue = rlglue.RLGlue(env, agent)


num_runs = 30
num_episodes = 50
cum_reward = np.zeros(num_episodes)

for run in tqdm(range(num_runs)):
    for ep in range(num_episodes):
        # run episode
        glue.episode(1000)
        cum_reward[ep] += glue.get_total_reward()
        glue.cleanup_episode()
        agent.epsilon *= 0.99
    learned_Q = agent.get_Q()
    see_action_values(learned_Q)
    glue.cleanup_run()
cum_reward /= float(num_runs)

np.save(
    f"experiments/feature_attainment/data_files/{env.name}_{agent.name}_average_return",
    cum_reward,
)
