import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


names = ["PPO_2room_doorA", "PPO_2room_doorB", "PPO_2room_doorC", "PPO_2room_doorD"]
plt.figure(figsize=(10, 6))
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.xlabel(r'$10^3$ Steps', fontsize=18)
plt.ylabel('Return', fontsize=18)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
# colors = {"PPO_2room_doorA": "#0077BB",
#           "PPO_2room_doorB": "#EE3377",
#           "PPO_2room_doorC": "#EE7733",
#           "PPO_2room_doorD": "#009988"}

for name in names:
    df = pd.read_csv(f"{name}.csv")
    if name == "PPO_2room_doorA":
        plt.plot(df['step'], df["ep_rew_mean"])
        print(len(df["step"]), len(df["ep_rew_mean"]))
    else:
        zeros = np.zeros(300)
        # append zeros to the first 300 steps
        plt.plot(np.append(zeros, df["ep_rew_mean"]))
        # print(len(np.append(zeros, df['step'])), len(np.append(zeros, df["ep_rew_mean"])))

        # print(len(df["step"]), len(df["ep_rew_mean"]))
        # plt.plot(df["step"], df["ep_rew_mean"])
plt.tight_layout()
plt.axvline(x=300, color='k', linestyle='--')
# plt.show()
plt.savefig(f"test.png")
# plt.savefig("PPO_2room_learningcurves.pdf")