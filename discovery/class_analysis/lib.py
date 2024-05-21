"""This module contains the classification analysis functions."""

from tkinter import W
from typing import Callable, Union, Type
from collections.abc import Iterable
import functools

import itertools
from sklearn import metrics

import numpy as np

import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from discovery.environments.custom_minigrids import TwoRoomEnv
import torch
import os
from stable_baselines3 import PPO

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

from discovery.class_analysis import datasources


def _to_tensor(feats):
    if isinstance(feats, list):
        X = torch.cat(feats, dim=0)
    elif isinstance(feats, torch.Tensor):
        X = feats
    else:
        X = torch.tensor(feats).float()
    return X


def train_classifier(
    clf,
    feats: Union[torch.Tensor, list[torch.Tensor]],
    labels: list,
    n_epochs=500,
    batch_size=32,
    test_size=0.0001,
    random_state=None,
    disable_tqdm=False,
):
    X = _to_tensor(feats)
    y = torch.tensor(labels).float()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    best_acc = -np.inf
    best_weights = None
    batch_start = torch.arange(
        0, len(X_train), batch_size
    )  # TODO: check if the last batch is included
    loss_fn = nn.BCELoss(
        reduction="none"
    )  # reduction='none' to get per-sample loss, not mean

    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    base_weight = torch.tensor(
        [1.0, num_neg / num_pos]
    )  # for weighted mean in loss calculation

    optimizer = optim.Adam(clf.parameters(), lr=0.0001)
    # TODO: collect positive examples, and concatenate them to each batch

    for epoch in range(n_epochs):
        clf.train()
        with tqdm.tqdm(
            batch_start, unit="batch", mininterval=0, disable=disable_tqdm
        ) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start : start + batch_size]
                y_batch = y_train[start : start + batch_size]
                # forward pass
                y_pred = clf(X_batch)
                y_batch = y_batch.unsqueeze(1)
                weight = torch.where(y_batch == 1, base_weight[1], base_weight[0])
                loss2 = loss_fn(y_pred, y_batch)
                final_loss = torch.mean(weight * loss2)
                # backward pass
                optimizer.zero_grad()
                final_loss.backward()
                # update weights
                optimizer.step()
                # print progress
                # TODO: the accuracy caculation is WRONG ------
                # see evaluate
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(loss=float(final_loss), acc=float(acc))
        # # evaluate accuracy at end of each epoch
        # clf.eval()
        # y_pred = clf(X_test)
        # acc = (y_pred.round() == y_test).float().mean()
        # acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(clf.state_dict())
    return best_acc


def predictions(clf, feats: Union[torch.Tensor, list[torch.Tensor]]):
    X = _to_tensor(feats)
    y_pred = clf(X)
    return y_pred


def evaluate(clf, feats, labels, print_results=False):
    y_pred = predictions(clf, feats)
    y = torch.tensor(labels).float()

    # acc_bad = (y_pred.round() == y).float().mean()
    y_pred_np = y_pred.detach().numpy().round()
    acc = metrics.accuracy_score(labels, y_pred_np)
    c_m = metrics.confusion_matrix(labels, y_pred_np)
    if print_results:
        print("Accuracy: ", acc)
        print("Confusion Matrix: ")
        print(c_m)
    return acc, c_m


def obs_to_feats_from_model(model, obss):

    def _map(obs_tensor):
        if model.__class__.__name__ == "DoubleDQN":
            x = model.policy.extract_features(
                obs_tensor, model.policy.q_net.features_extractor
            )
        elif model.__class__.__name__ == "PPO":
            x = model.policy.extract_features(obs_tensor)
        return x

    try:
        with torch.no_grad():
            feats = _map(obss)
    except Exception as e:
        print(
            "Could not process the full tensor in one go, will try to "
            "process one-by-one. Original error:\n",
            e,
        )
        with torch.no_grad():
            feats = [_map(obs) for obs in obss]
    return feats


def process_saved_model(data_manager: datasources.DataSource, model_path: str) -> dict:
    model = PPO.load(model_path)

    def obs_to_feats(obss):
        preproc_obss = data_manager.obs_preprocessor(obss)
        tensors = obs_as_tensor(preproc_obss, model.device)
        return obs_to_feats_from_model(model, tensors)

    return process_model(data_manager, obs_to_feats)


def process_model(
    data_manager: datasources.DataSource,
    obs_to_feats: Callable[[Iterable], list],
) -> dict:
    # def process_model(data_manager_cls, model_path: str):
    """Process a single model."""
    obss, images, labels = data_manager.get_data()
    feats = obs_to_feats(obss)
    feat_dim = feats[0].shape[-1]
    classifiers = {
        "linear": sg_detection.LinearClassifier(input_size=feat_dim),
        "nonlinear": sg_detection.NonLinearClassifier(
            input_size=feat_dim, hidden_size=64
        ),
    }
    results = {}
    for clf_name, clf in classifiers.items():
        unused_best_acc = train_classifier(clf, feats, labels, disable_tqdm=True)
        acc, conf_mat = evaluate(clf, feats, labels)
        details = {
            "classifier": clf,
            "obs_to_feats": obs_to_feats,
        }
        results[clf_name] = (float(acc), conf_mat, details)
    return results
