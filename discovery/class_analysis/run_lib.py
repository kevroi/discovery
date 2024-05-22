"""A bunch of helpers for run.py, just to clear up space in there."""

import os
import pickle
import numpy as np

from discovery.class_analysis.datatypes import (
    Setting,
    Data,
    EnvName,
    ModelType,
    BaseStats,
    BaseTrainTestStats,
    BinaryClassStats,
    AllTrainTestStats,
)


def make_path_relative(path: str) -> str:
    """Returns the path relative to the current working directory/project root."""
    # Not really tested.
    if path.startswith("/"):
        return path[len(os.getcwd()) + 1 :]
    return path


def scantree(abs_path, exclude_dirs_abs_paths=None):
    """Recursively yield DirEntry objects for given directory."""
    # Modified from https://stackoverflow.com/a/33135143
    if exclude_dirs_abs_paths is None:
        exclude_dirs_abs_paths = []
    for entry in os.scandir(abs_path):
        if entry.is_dir(follow_symlinks=False):
            if entry.path in exclude_dirs_abs_paths:
                print("Skipping excluded dir:", entry.path)
                continue
            yield from scantree(entry.path, exclude_dirs_abs_paths)
        else:
            yield entry


def load_existing_results(result_path: str):
    """Returns the existing results, or an empty dict if there are none."""
    try:
        with open(result_path, "rb") as f:
            existing_results = pickle.load(f)
    except FileNotFoundError:
        existing_results = {}
        print(
            "There is no results file at",
            os.path.join(os.getcwd(), result_path),
            "\nStarting a new results file!",
        )
    return existing_results


def append_or_create_V2(
    all_results: dict[Setting, AllTrainTestStats],
    setting: Setting,
    wandb_id: str,
    cur_results: BaseTrainTestStats,
):
    """Appends the results to the data for the setting, or creates a new entry if needed."""

    def update_binary_class_stats(bcs: BinaryClassStats, bs: BaseStats, num_runs):
        bcs.accs.append(bs.acc)
        bcs.sg_accs.append(bs.sg_acc)
        bcs.non_sg_accs.append(bs.non_sg_acc)
        bcs.conf_matrices.append(bs.conf_mat)
        bcs.acc_mean = np.mean(bcs.accs)
        bcs.acc_std_err = np.std(bcs.accs) / np.sqrt(num_runs)
        bcs.sg_acc_mean = np.mean(bcs.sg_accs)
        bcs.sg_acc_std_err = np.std(bcs.sg_accs) / np.sqrt(num_runs)
        bcs.non_sg_acc_mean = np.mean(bcs.non_sg_accs)
        bcs.non_sg_acc_std_err = np.std(bcs.non_sg_accs) / np.sqrt(num_runs)

    if setting in all_results:
        all_train_test_stats = all_results[setting]
    else:
        all_train_test_stats = AllTrainTestStats(
            wandb_ids=[],
            num_runs=0,
            lin_train=BinaryClassStats([], [], [], [], 0, 0, 0, 0, 0, 0),
            lin_test=BinaryClassStats([], [], [], [], 0, 0, 0, 0, 0, 0),
            nonlin_train=BinaryClassStats([], [], [], [], 0, 0, 0, 0, 0, 0),
            nonlin_test=BinaryClassStats([], [], [], [], 0, 0, 0, 0, 0, 0),
        )
        all_results[setting] = all_train_test_stats

    all_train_test_stats.wandb_ids.append(wandb_id)
    num_runs = all_train_test_stats.num_runs + 1
    all_train_test_stats.num_runs = num_runs

    update_binary_class_stats(
        all_train_test_stats.lin_train, cur_results.lin_train, num_runs
    )
    update_binary_class_stats(
        all_train_test_stats.lin_test, cur_results.lin_test, num_runs
    )
    update_binary_class_stats(
        all_train_test_stats.nonlin_train, cur_results.nonlin_train, num_runs
    )
    update_binary_class_stats(
        all_train_test_stats.nonlin_test, cur_results.nonlin_test, num_runs
    )


def append_or_create(
    all_results: dict[Setting, Data],
    setting: Setting,
    wandb_id: str,
    cur_results: dict[str, tuple[BaseStats, BaseStats, dict]],
):
    """Appends the results to the data for the setting, or creates a new entry."""
    lin_acc, lin_sg_acc, lin_non_sg_acc, lin_conf_mat = cur_results["linear"][
        0
    ]  # 0 is for train data.
    nonlin_acc, nonlin_sg_acc, nonlin_non_sg_acc, nonlin_conf_mat = (
        cur_results["nonlinear"]
    )[
        0
    ]  # 0 is for train data.
    if setting in all_results:
        data = all_results[setting]
        data.num_runs += 1
        data.wandb_ids.append(wandb_id)
        data.lin_accuracies.append(lin_acc)
        data.lin_sg_accuracies.append(lin_sg_acc)
        data.lin_non_sg_accuracies.append(lin_non_sg_acc)
        data.lin_conf_matrices.append(lin_conf_mat)
        data.lin_acc_mean = np.mean(data.lin_accuracies)
        data.lin_acc_std_err = np.std(data.lin_accuracies) / np.sqrt(data.num_runs)
        data.lin_sg_acc_mean = np.mean(data.lin_sg_accuracies)
        data.lin_sg_acc_std_err = np.std(data.lin_sg_accuracies) / np.sqrt(
            data.num_runs
        )
        data.lin_non_sg_acc_mean = np.mean(data.lin_non_sg_accuracies)
        data.lin_non_sg_acc_std_err = np.std(data.lin_non_sg_accuracies) / np.sqrt(
            data.num_runs
        )

        data.nonlin_accuracies.append(nonlin_acc)
        data.nonlin_conf_matrices.append(nonlin_conf_mat)
        data.nonlin_acc_mean = np.mean(data.nonlin_accuracies)
        data.nonlin_acc_std_err = np.std(data.nonlin_accuracies) / np.sqrt(
            data.num_runs
        )
        data.nonlin_sg_acc_mean = np.mean(data.nonlin_sg_accuracies)
        data.nonlin_sg_acc_std_err = np.std(data.nonlin_sg_accuracies) / np.sqrt(
            data.num_runs
        )
        data.nonlin_non_sg_acc_mean = np.mean(data.nonlin_non_sg_accuracies)
        data.nonlin_non_sg_acc_std_err = np.std(
            data.nonlin_non_sg_accuracies
        ) / np.sqrt(data.num_runs)
    else:
        all_results[setting] = Data(
            wandb_ids=[wandb_id],
            num_runs=1,
            lin_accuracies=[lin_acc],
            lin_sg_accuracies=[lin_sg_acc],
            lin_non_sg_accuracies=[lin_non_sg_acc],
            lin_conf_matrices=[lin_conf_mat],
            lin_acc_mean=lin_acc,
            lin_acc_std_err=0.0,
            lin_sg_acc_mean=lin_sg_acc,
            lin_sg_acc_std_err=0.0,
            lin_non_sg_acc_mean=lin_non_sg_acc,
            lin_non_sg_acc_std_err=0.0,
            nonlin_accuracies=[nonlin_acc],
            nonlin_sg_accuracies=[nonlin_sg_acc],
            nonlin_non_sg_accuracies=[nonlin_non_sg_acc],
            nonlin_conf_matrices=[nonlin_conf_mat],
            nonlin_acc_mean=nonlin_acc,
            nonlin_acc_std_err=0.0,
            nonlin_sg_acc_mean=nonlin_sg_acc,
            nonlin_sg_acc_std_err=0.0,
            nonlin_non_sg_acc_mean=nonlin_non_sg_acc,
            nonlin_non_sg_acc_std_err=0.0,
        )
