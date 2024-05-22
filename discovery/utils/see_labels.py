import numpy as np
import h5py
import matplotlib.pyplot as plt

x = np.load("datasets/AAD/clean/SeaquestNoFrameskip-v4/episode(1)_labels.npy")
print(x.shape)
# x[764] = 2
# x[953:965] = 1
# print(x[764])
# np.save("datasets/AAD/clean/SeaquestNoFrameskip-v4/episode(1)_labels.npy", x)


# ENV = "SeaquestNoFrameskip-v4"
# label_keys = ["0", "1", "2", "3"]  # air = 1, shoot = 2, dive = 3

# episode = 1

# if episode is not None:
#     path = f"datasets/AAD/clean/{ENV}/episode({episode}).hdf5"
# else:
#     path = f"datasets/AADclean/{ENV}/episode.hdf5"

# with h5py.File(path, "r") as f:
#     state = f["state"][...]

# fig, ax = plt.subplots()
# ax.imshow(state[765])
# plt.show()
