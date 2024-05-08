import sys
import numpy as np
import pickle
import copy

import utils.rlglue as rlglue
from environments.environment import ToyEnvironment, GridEnvironment, I_MazeEnvironment, phi1Environment, phi2Environment
from agents.QLearning import QAgent, QAgent_phi
from agents.QLearning import OptionExploreQAgent
from eigenoptions.options import Options

explore_env = ToyEnvironment('2room')
max_row, max_col = explore_env.get_grid_dimension() # get dimension of the environment
# explore_agent = OptionExploreQAgent(max_row=max_row, max_col=max_col)
# phi_1 = [(1, 1), (1, 3)]
# phi_2 = [(1, 3), (1, 4)]
# explore_agent = QAgent_phi(max_row=max_row, max_col=max_col, phi_sg=[(1, 3), (1, 4)])
# explore_agent.set_alpha(0.1)
# explore_agent.set_discount(0.9)
# explore_agent.set_epsilon(0.0)
# explore_glue = rlglue.RLGlue(explore_env, explore_agent)
reward_env = ToyEnvironment('2room')
max_row, max_col = reward_env.get_grid_dimension() # get dimension of the environment
reward_agent = QAgent_phi(max_row=max_row, max_col=max_col, phi_sg=[(1,3), (0,4)])
# reward_agent = QAgent(max_row=max_row, max_col=max_col)
reward_agent.set_alpha(0.05)
reward_agent.set_epsilon(0.1)
reward_agent.set_discount(0.9)
reward_glue = rlglue.RLGlue(reward_env, reward_agent)
# Experiment
np.set_printoptions(precision=2)

# Experiment parameter
num_runs = 100
num_episodes = 100
num_options = 256

# Starting from the agent with primitive actions, we incrementally add options
# in explore_agent
results = np.zeros((num_options+1, num_episodes))
steps = np.zeros((num_options+1, num_episodes))
ret_mat = np.zeros((2, num_runs, num_episodes))
steps_mat = np.zeros((2, num_runs, num_episodes))

cum_reward = np.zeros(num_episodes)
steps_to_goal = np.zeros(num_episodes)
for run in range(num_runs):
    for ep in range(num_episodes):
        # run episode
        # explore_glue.episode(100)
        # learned_Q = explore_agent.get_Q()
        # reward_agent.set_Q(learned_Q)
        reward_glue.episode(50)
        # cum_reward[ep] += reward_glue.get_total_reward()
        # steps_to_goal[ep] += reward_glue.num_steps
        ret_mat[0, run, ep] = reward_glue.get_total_reward()
        steps_mat[0, run, ep] = reward_glue.num_steps
        reward_glue.cleanup_episode()
    # explore_glue.cleanup_run()
# cum_reward /= float(num_runs)
# steps_to_goal /= float(num_runs)

# reward_agent.set_epsilon(0.0)

# cum_reward = np.zeros(num_episodes)
# for run in range(num_runs):
#     for ep in range(num_episodes):
#         # run episode
#         # explore_glue.episode(100)
#         # learned_Q = explore_agent.get_Q()
#         # reward_agent.set_Q(learned_Q)
#         reward_glue.episode(100)
#         cum_reward[ep] += reward_glue.get_total_reward()
#         reward_glue.cleanup_episode()
#     # explore_glue.cleanup_run()
# cum_reward /= float(num_runs)
# results[0] = cum_reward
# steps[0] = steps_to_goal

reward_env = ToyEnvironment('2room')
max_row, max_col = reward_env.get_grid_dimension() # get dimension of the environment
reward_agent = QAgent_phi(max_row=max_row, max_col=max_col, phi_sg=[(1,3), (1,2)])
# reward_agent = QAgent(max_row=max_row, max_col=max_col)
reward_agent.set_alpha(0.05)
reward_agent.set_epsilon(0.1)
reward_agent.set_discount(0.9)
reward_glue = rlglue.RLGlue(reward_env, reward_agent)

# cum_reward = np.zeros(num_episodes)
# steps_to_goal = np.zeros(num_episodes)
for run in range(num_runs):
    for ep in range(num_episodes):
        # run episode
        # explore_glue.episode(100)
        # learned_Q = explore_agent.get_Q()
        # reward_agent.set_Q(learned_Q)
        reward_glue.episode(50)
        ret_mat[1, run, ep] = reward_glue.get_total_reward()
        steps_mat[1, run, ep] = reward_glue.num_steps
        reward_glue.cleanup_episode()
    # explore_glue.cleanup_run()
# cum_reward /= float(num_runs)
# steps_to_goal /= float(num_runs)
# results[1] = cum_reward
# steps[1] = steps_to_goal
        

# plot avg return aganisnt episode, with standard err
import matplotlib.pyplot as plt

# get avg return and std err across runs
avg_ret = np.mean(ret_mat, axis=1)
std_err = np.std(ret_mat, axis=1) / np.sqrt(num_runs)
avg_steps = np.mean(steps_mat, axis=1)
std_steps = np.std(steps_mat, axis=1) / np.sqrt(num_runs)

#plot both lines
plt.figure()
x_legend = range(num_episodes)
graph_agent_0, = plt.plot(x_legend, avg_ret[0], label="primitive actions", color='#E61C17') # red
graph_agent_2, = plt.plot(x_legend, avg_ret[1], label="2 option", color='#26B812') # green
plt.fill_between(x_legend, avg_ret[0]-std_err[0], avg_ret[0]+std_err[0], alpha=0.3, color='#E61C17')
plt.fill_between(x_legend, avg_ret[1]-std_err[1], avg_ret[1]+std_err[1], alpha=0.3, color='#26B812')
plt.xlabel('Episodes')
plt.ylabel('Average Return')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.ylim(0.8, 1.01)
plt.savefig(f'experiments/average_return/plots/2room_average_return.pdf', dpi=300)

# plot avg steps
plt.figure()
x_legend = range(num_episodes)
graph_agent_0, = plt.plot(x_legend, avg_steps[0], label="primitive actions", color='#E61C17')
graph_agent_2, = plt.plot(x_legend, avg_steps[1], label="2 option", color='#26B812')
plt.fill_between(x_legend, avg_steps[0]-std_steps[0], avg_steps[0]+std_steps[0], alpha=0.3, color='#E61C17')
plt.fill_between(x_legend, avg_steps[1]-std_steps[1], avg_steps[1]+std_steps[1], alpha=0.3, color='#26B812')
plt.xlabel('Episodes')
plt.ylabel('Average Steps')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.ylim(0, 20)
plt.savefig(f'experiments/average_return/plots/2room_average_steps.pdf', dpi=300)



# print(results)

# np.save(f'experiments/average_return/data_files/{reward_env.name}_average_return')
# np.save(f'experiments/average_return/data_files/{reward_env.name}_average_steps', steps)