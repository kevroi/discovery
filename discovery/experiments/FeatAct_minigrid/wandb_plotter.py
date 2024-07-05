import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.utils import resample


def calculate_bootstrapped_ci(data, n_bootstrap=1000, ci_level=0.95):
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        bootstrapped_sample = resample(data, replace=True)
        bootstrapped_means.append(np.mean(bootstrapped_sample))

    alpha = (1 - ci_level) / 2
    lower_ci = np.percentile(bootstrapped_means, alpha * 100)
    upper_ci = np.percentile(bootstrapped_means, (1 - alpha) * 100)

    return lower_ci, upper_ci


def plot_wandb_data(
    project_name="two_rooms_realmultitask",
    env_name="TwoRoomEnv",
    learner="PPO",
    activation="relu",
    # feat_dim=40,
    metric="rollout/ep_rew_mean",
    fig=None,
    color=None,
    ci="normal",
):

    df = pd.read_csv(
        f"discovery/experiments/FeatAct_minigrid/wandb_export_data/{project_name}_{env_name}_{learner}_{activation}.csv"
    )
    mean_rewards = df.iloc[:, 1:].mean(
        axis=1
    )  # Ignore the first column (index) and calculate mean
    std_rewards = df.iloc[:, 1:].std(
        axis=1
    )  # Ignore the first column (index) and calculate standard deviation

    if ci == "normal":
        # Calculate the 95% confidence interval
        n = df.shape[1] - 1  # Number of columns excluding the index column
        ci = 1.96 * (std_rewards / np.sqrt(n))
        lower_ci = mean_rewards - ci
        upper_ci = mean_rewards + ci
    elif ci == "bootstrap":
        # Calculate the bootstrapped 95% confidence interval
        lower_ci, upper_ci = calculate_bootstrapped_ci(mean_rewards.values)

    # Plot the average line with the shaded region representing the 95% confidence interval
    if fig is None:
        fig = plt.figure(figsize=(10, 6))
    plt.plot(mean_rewards.index, mean_rewards, label=activation, color=color)
    plt.fill_between(mean_rewards.index, lower_ci, upper_ci, color=color, alpha=0.3)

    return fig


if __name__ == "__main__":
    colors = {
        "relu": "#0077BB",
        "crelu": "#EE7733",
        "fta": "#EE3377",
        "lrelu": "#009988",
    }
    fig = None
    activations = [
        "relu",
        "fta",
        # "crelu",
    ]
    for act in activations:
        fig = plot_wandb_data(activation=act, fig=fig, color=colors[act])
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    posiitons = [0, 100, 200, 300, 400, 500]
    labels = ["0", "200k", "400k", "600k", "800k", "1M"]
    plt.xticks(posiitons, labels, fontsize=18)
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Return", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, 1)
    plt.grid(False)
    # plt.legend(fontsize=18, loc="upper left")
    # plt.show()
    plt.savefig("plots/learning_curves/TwoRooms_multitask_diff_acts.pdf")
