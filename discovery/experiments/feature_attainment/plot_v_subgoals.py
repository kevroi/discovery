import numpy as np
import matplotlib.pyplot as plt
import sys

env_name = "grid"
env_name = "2room_lava"
agent_name = "Sarsa(lambda)"
agent_name = "Sarsa(lambda)_FeatAtt"
# env_name = "4room"

# if __name__ == "__main__":
results = np.load(
    f"experiments/feature_attainment/data_files/{env_name}_{agent_name}_hallway_vf.npy"
)
print(np.shape(results))

plt.show()
x_legend = range(len(results[:]))
(graph_agent_0,) = plt.plot(x_legend, results, label=agent_name)
# graph_agent_2, = plt.plot(x_legend, results[2][:], label="2 option")
# graph_agent_4, = plt.plot(x_legend, results[4][:], label="4 options")
# graph_agent_8, = plt.plot(x_legend, results[8][:], label="8 options")
# graph_agent_64, = plt.plot(x_legend, results[64][:], label="64 options")
# graph_agent_128, = plt.plot(x_legend, results[128][:], label="128 options")
# graph_agent_200, = plt.plot(x_legend, results[200][:], label="200 options")

# plt.axhline(y=-18, color='purple', linestyle='--', label='Optimal')


plt.legend(handles=[graph_agent_0])
plt.xlabel("Episodes")
plt.ylabel("Average return")
plt.title(f"{env_name}")
plt.show()
# plt.savefig(f'experiments/average_return/plots/{env_name}_average_return.png', dpi=300)
