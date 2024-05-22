import numpy as np
import h5py
import matplotlib.pyplot as plt


def on_key(event):
    global current_frame
    if event.key in label_keys:
        print(f"Label for frame {current_frame}: {event.key}")
        labels[current_frame] = int(event.key)
        current_frame += 1

        if current_frame < num_images:
            ax.imshow(state[current_frame])
            fig.canvas.draw()
        else:
            plt.close(fig)


ENV = "SeaquestNoFrameskip-v4"
label_keys = ["0", "1", "2", "3"]  # air = 1, shoot = 2, dive = 3

episode = 59

if episode is not None:
    path = f"datasets/AAD/clean/{ENV}/episode({episode}).hdf5"
else:
    path = f"datasets/AADclean/{ENV}/episode.hdf5"

with h5py.File(path, "r") as f:
    state = f["state"][...]

num_images = state.shape[0]
labels = np.zeros(num_images, dtype=int)
current_frame = 0

# TODO fix this so it doesnt slow down and plot every frame
fig, ax = plt.subplots()
ax.imshow(state[current_frame])
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()

np.save(f"datasets/AAD/clean/{ENV}/episode({episode})_labels.npy", labels)
