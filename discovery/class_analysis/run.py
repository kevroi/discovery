import functools
from multiprocessing import process
from typing import Optional

import argparse
import os
from dataclasses import dataclass
import numpy as np
import tqdm
import pickle
from discovery.utils import filesys
from discovery.utils import sg_detection
from discovery.class_analysis import lib

from stable_baselines3 import PPO


_RESULT_STORE = "discovery/class_analysis/results.pkl"

parser = argparse.ArgumentParser()
# The config file and things we want to sweep over (overwriting the config file)
# CAREFUL: These default args also overwrite the config file
parser.add_argument(
    "--clear_results",
    action="store_true",  # set to false if we do not pass this argument
    help="Clears all previous results.",
)


@dataclass
class Model:
    multitask: bool
    model_type: str
    wandb_id: str
    model_path: str
    train_accuracy: Optional[float]
    test_accuracy: Optional[float]
    train_confusion_matrix: Optional[np.array]
    test_confusion_matrix: Optional[np.array]

    def __init__(self, multitask, model_type, wandb_id, model_path):
        self.multitask = multitask
        self.model_type = model_type
        self.wandb_id = wandb_id
        self.model_path = model_path
        self.train_accuracy = None
        self.test_accuracy = None
        self.train_confusion_matrix = None
        self.test_confusion_matrix = None


minigrid_models = {
    "multitask_cnn": Model(
        multitask=True,
        model_type="cnn",
        wandb_id="5i6lt53x",
        model_path="experiments/FeatAct_minigrid/models/PPO_TwoRoomEnv_5i6lt53x.zip",
    ),
}

climbing_models = {}

atari_models = {}

model_types = {
    "minigrid": (minigrid_models, lib.MiniGridData),
    "climbing": (climbing_models, None),
    "atari": (atari_models, None),
}


def main():
    args = parser.parse_args()

    filesys.set_directory_in_project()

    if not args.clear_results:
        # Load the results we already have.
        try:
            with open(_RESULT_STORE, "rb") as f:
                existing_results = pickle.load(f)
        except FileNotFoundError:
            existing_results = {}
            print(
                "There is no results file at", os.path.join(os.getcwd(), _RESULT_STORE)
            )
    else:
        existing_results = {}

    # Iterating over model types.
    all_results = {}  # Indexed model type
    for mt_idx, model_type in enumerate(model_types.keys()):
        coll_of_models, data_proc = model_types[model_type]
        print(f"Model type {mt_idx}/{len(model_types)}: {model_type}")
        if model_type not in existing_results:
            existing_results[model_type] = {}
        cur_results = existing_results[model_type]
        for name, model_desc in tqdm.tqdm(coll_of_models.items()):
            if name in cur_results:
                print("Skipping", name)
                # TODO: in the future, check if it has all the data.
            else:
                print("Processing", name)
                cur_results[name] = process_model(data_proc, model_desc)

    with open(_RESULT_STORE, "wb") as f:
        pickle.dump(existing_results, f)


def process_model(data_proc, model_desc: Model):
    """Process a single model."""
    data_mgr = data_proc()
    model = PPO.load(model_desc.model_path)
    obs_to_feats = functools.partial(lib.obs_to_feats, model)
    obss, images, labels = data_mgr.get_data()
    feats = obs_to_feats(obss)
    clf = sg_detection.LinearClassifier(input_size=32)
    unused_best_acc = lib.train_classifier(clf, feats, labels)
    acc, conf_mat = lib.evaluate(clf, feats, labels)
    return acc, conf_mat


if __name__ == "__main__":
    main()
