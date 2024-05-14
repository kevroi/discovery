import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.activations import *
import matplotlib.cm as cm

# Number of random tensors to generate
num_tensors = 5
# Set the seed for reproducibility
torch.manual_seed(0)

# Generate random tensors in R^2
tensors = torch.randn(num_tensors, 2)

# Plot the tensors with different colors
colors = cm.rainbow(torch.linspace(0, 1, num_tensors))
plt.figure(figsize=(6, 6))
plt.grid(False)
for i in range(num_tensors):
    plt.arrow(0, 0, tensors[i, 0], tensors[i, 1], color=colors[i], head_width=0.1)
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.title("Vectors in $\mathbb{R}^2$ pre-activation", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.show()
plt.savefig("plots/activations/activations_pre.pdf")


# Create a figure and axes for each pair of entries
activation = "crelu"
if activation == "relu":
    transformed_tensors = F.relu(tensors)
elif activation == "crelu":
    transformed_tensors = CReLU()(tensors)
elif activation == "fta":
    transformed_tensors = FTA(lower_limit=-2, upper_limit=2, delta=0.5)(tensors)
elif activation == "lrelu":
    transformed_tensors = F.leaky_relu(tensors)

transformed_tensors_np = transformed_tensors.detach().numpy()
# Create subplots
num_subplots = transformed_tensors.shape[1] // 2
limits = 2.5
fig, axs = plt.subplots(num_subplots, 1, figsize=(3, 3 * num_subplots), squeeze=False)
for i in range(num_subplots):
    for j in range(num_tensors):
        axs[i, 0].arrow(
            0,
            0,
            transformed_tensors_np[j, 2 * i],
            transformed_tensors_np[j, 2 * i + 1],
            color=colors[j],
            head_width=0.1,
        )
    axs[i, 0].set_xlabel(f"$\phi_{{{2*i}}}$")
    axs[i, 0].set_ylabel(f"$\phi_{{{2*i+1}}}$")
    axs[i, 0].set_xlim(-limits, limits)
    axs[i, 0].set_ylim(-limits, limits)
# Adjust layout and display the plot
plt.tight_layout()
# plt.show()
plt.savefig(f"plots/activations/post_activations_{activation}.pdf")
