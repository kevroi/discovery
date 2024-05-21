from typing import Optional

import abc
import cv2
import h5py
import itertools
from sklearn.metrics import confusion_matrix

import numpy as np

import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from discovery.environments.custom_minigrids import TwoRoomEnv
import torch
import os
from stable_baselines3 import PPO
from discovery.experiments.FeatAct_minigrid import helpers as minigrid_helpers


from discovery.utils.feat_extractors import (
    ClimbingFeatureExtractor,
    MinigridFeaturesExtractor,
)

from stable_baselines3.common.utils import obs_as_tensor

from discovery.utils import sg_detection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy
import tqdm
import torch.nn as nn
import torch.optim as optim


class DataSource(abc.ABC):

    @abc.abstractmethod
    def get_data(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def obs_preprocessor(self, obs):
        raise NotImplementedError()

    @abc.abstractmethod
    def visualize(self, clf, obs_to_feats_fn):
        raise NotImplementedError()


class MiniGridData(DataSource):

    def __init__(self, variants):
        obss, images, labels, coords_seq, dirs_seq = self._create_dataset(variants)
        self.obss = np.stack(obss)
        self.images = np.stack(images)
        self.labels = np.stack(labels)
        self.coords_seq = coords_seq
        self.dirs_seq = dirs_seq

    def obs_preprocessor(self, obs):
        return minigrid_helpers.pre_process_obs_no_tensor(obs)

    def get_data(self):
        # return self.obss, self.images, self.labels, self.coords_seq, self.dirs_se
        return self.obss, self.images, self.labels

    def visualize(self, clf, obs_to_feats_fn):
        feats = obs_to_feats_fn(self.obss)
        X = torch.cat(feats, dim=0)
        y_pred = clf(X)

        torch.set_printoptions(linewidth=150, precision=2)
        base = torch.ones((4, 7, 14)) * torch.nan
        for idx, (coord, dir_) in enumerate(zip(self.coords_seq, self.dirs_seq)):
            base[dir_, coord[1], coord[0]] = y_pred[idx].item()
        for i in base:
            print(i)
        torch.set_printoptions()  # Resets them.

    def _make_env_at_pos(self, position, direction, variant):
        env = TwoRoomEnv(
            render_mode="rgb_array",
            agent_start_pos=position,
            agent_start_dir=direction,
            hallway_pos=(variant, 7),
        )
        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        env = Monitor(env)
        return env

    def _create_dataset(self, variants):
        hallways = [(7, v) for v in variants]
        xs = range(1, 14)
        ys = range(1, 7)
        xys = itertools.product(xs, ys)
        coords_seq = []
        dirs_seq = []
        dirs = range(4)
        all_data = itertools.product(xys, dirs, variants)
        images = []
        obss = []
        labels = []
        for pos_, dir_, var_ in all_data:
            env = self._make_env_at_pos(position=pos_, direction=dir_, variant=var_)
            try:
                # obss.append(env.reset())
                obss.append(env.reset()[0])  # Just the observation.
                images.append(env.render())
                labels.append(bool(pos_ == (7, var_)))
                coords_seq.append(pos_)
                dirs_seq.append(dir_)
            except AssertionError:
                # print("bad place:", pos_)
                pass

        return obss, images, labels, coords_seq, dirs_seq


class SeaquestData(DataSource):

    def __init__(
        self, obs_filepath: str, label_filepath: str, cutoff: Optional[int] = None
    ):
        with h5py.File(obs_filepath, "r") as f:
            image_sequence = f["state"][...]
        labels = np.load(label_filepath)

        if cutoff is not None:
            image_sequence = image_sequence[:cutoff]
            labels = labels[:cutoff]
        self.obss = _pre_process_atari(image_sequence)  # pre_processed_states =
        labels = _prerocess_labels(labels)
        self.labels = _stack_labels(labels)  # stacked_labels =
        self.images = image_sequence

    def get_data(self):
        # return self.obss, self.images, self.labels, self.coords_seq, self.dirs_se
        return self.obss, self.images, self.labels

    def obs_preprocessor(self, obs):
        return obs

    def visualize(self, clf, obs_to_feats_fn):
        raise NotImplementedError()


def _pre_process_atari(dataset: np.ndarray):
    num_images = dataset.shape[0]
    preprocessed_images = np.zeros((num_images, 84, 84), dtype=np.uint8)

    for i in range(num_images):
        preprocessed_images[i] = cv2.resize(
            cv2.cvtColor(dataset[i], cv2.COLOR_RGB2GRAY), (84, 84)
        )

        # Stack frames
        stacked_images = np.zeros((num_images - 3, 4, 84, 84), dtype=np.uint8)
        for i in range(num_images - 3):
            stacked_images[i] = np.stack(
                [
                    preprocessed_images[i],
                    preprocessed_images[i + 1],
                    preprocessed_images[i + 2],
                    preprocessed_images[i + 3],
                ]
            )

    return stacked_images


def _prerocess_labels(labels):
    # turn all 1s and 3s into 0s
    labels[labels == 1] = 0
    labels[labels == 3] = 0
    # turn all 2s into 1s
    labels[labels == 2] = 1
    return labels


def _stack_labels(labels):
    stacked_labels = np.zeros((labels.shape[0] - 3, 1), dtype=np.uint8)
    for i in range(labels.shape[0] - 3):
        stacked_labels[i] = labels[i + 3]
    return np.squeeze(stacked_labels)
