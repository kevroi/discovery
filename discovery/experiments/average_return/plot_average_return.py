import numpy as np
import matplotlib.pyplot as plt
import sys

env_name = "2room"
# env_name = "4room"

# if __name__ == "__main__":
results = np.load(f'experiments/average_return/data_files/{env_name}_average_return.npy')

plt.show()
x_legend = range(len(results[0][:]))
print(results[0][:])
graph_agent_0, = plt.plot(x_legend, results[0][:], label="primitive actions", color='#E61C17')
graph_agent_2, = plt.plot(x_legend, results[1][:], label="2 option", color='#26B812')
# graph_agent_4, = plt.plot(x_legend, results[4][:], label="4 options")
# graph_agent_8, = plt.plot(x_legend, results[8][:], label="8 options")
# graph_agent_64, = plt.plot(x_legend, results[64][:], label="64 options")
# graph_agent_128, = plt.plot(x_legend, results[128][:], label="128 options")
# graph_agent_200, = plt.plot(x_legend, results[200][:], label="200 options")

# plt.legend(handles=[graph_agent_0, graph_agent_2, graph_agent_4, graph_agent_8, graph_agent_64, graph_agent_128])
plt.xlabel('Episodes')
plt.ylabel('Average Return')
# plt.title(f'{env_name}')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
# plt.show()
plt.ylim(0, 1.01)
# plt.savefig(f'experiments/average_return/plots/{env_name}_average_return_NormalGoal.pdf', dpi=300)
plt.show()



results = np.load(f'experiments/average_return/data_files/{env_name}_average_steps.npy')

plt.show()
x_legend = range(len(results[0][:]))
print(results[0][:])
graph_agent_0, = plt.plot(x_legend, results[0][:], label="primitive actions", color='#E61C17')
graph_agent_2, = plt.plot(x_legend, results[1][:], label="2 option", color='#26B812')

# plt.legend(handles=[graph_agent_0, graph_agent_2, graph_agent_4, graph_agent_8, graph_agent_64, graph_agent_128])
plt.xlabel('Episodes')
plt.ylabel('Average Steps')
# plt.title(f'{env_name}')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
# plt.show()
plt.ylim(0, 50)
# plt.savefig(f'experiments/average_return/plots/{env_name}_average_return_NormalGoal.pdf', dpi=300)
plt.show()
