import sys
import numpy as np
import pickle
import copy
from tqdm import tqdm

import utils.rlglue as rlglue
from environments.environment import ToyEnvironment, GridEnvironment, I_MazeEnvironment
from agents.TD_lambda import TDLambdaAgent
from agents.Sarsa_lmbda import SarsaLmbdaAgent, SarsaLambdaFeatAtt
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
agent = SarsaLambdaFeatAtt(max_row=max_row, max_col=max_col, subgoals=[(2, 6)])
agent.set_alpha(0.1)
agent.set_discount(0.99)
agent.set_epsilon(0.1)
agent.set_lmbda(0.0)
agent.set_subgoals([(2, 6)])  # 2rooms
glue = rlglue.RLGlue(env, agent)


num_runs = 10
num_episodes = 40
cum_reward = np.zeros(num_episodes)
v_hallway = np.zeros(num_episodes)
q_hallway = np.zeros(num_episodes)
action_set = ["→", "←", "↓", "↑"]

for run in tqdm(range(num_runs)):
    for ep in range(num_episodes):
        # run episode
        glue.episode(1000)

        cum_reward[ep] += glue.get_total_reward()
        v_hallway[ep] += agent.get_v_subgoals([2, 0])[0]
        q_hallway[ep] += np.max(agent.get_q_subgoals([2, 0])[0])
        # print(agent.get_v_subgoals([2,0])[0])
        agent.alpha *= 0.999
        glue.cleanup_episode()
    # learned_Q = agent.get_Q()
    # see_action_values(learned_Q)
    # print("V_hallway: ")
    # print(np.round(agent.subgoal_vfs[0]["V"], decimals=1))
    # print("Q_hallway: ")
    # see_action_values(agent.subgoal_vfs[0]["Q"])
    # print(np.argmax(agent.subgoal_vfs[0]["Q"], axis=2))
    glue.cleanup_run()
cum_reward /= float(num_runs)
v_hallway /= float(num_runs)


np.save(
    f"experiments/feature_attainment/data_files/{env.name}_{agent.name}_average_return",
    cum_reward,
)
np.save(
    f"experiments/feature_attainment/data_files/{env.name}_{agent.name}_hallway_vf",
    v_hallway,
)
# np.save(f'experiments/feature_attainment/data_files/{env.name}_{agent.name}_hallway_vf', q_hallway)
