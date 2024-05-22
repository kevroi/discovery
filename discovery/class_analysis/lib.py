"""This module contains the classification analysis functions."""

from typing import Callable, Union
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
from discovery.class_analysis.datatypes import BaseStats


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
    feats: torch.Tensor,
    labels: torch.Tensor,
    n_epochs=500,
    batch_size=32,
    disable_tqdm=False,
):
    """All data is used for training, and test_size is ignored."""
    X_train = feats
    y_train = labels

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
                acc = metrics.accuracy_score(y_batch, y_pred.detach().numpy().round())
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

    # acc_bad = (y_pred.round() == y).float().mean()
    y_pred_np = y_pred.detach().numpy().round()
    acc = metrics.accuracy_score(labels, y_pred_np)
    c_m = metrics.confusion_matrix(labels, y_pred_np, labels=[0, 1])
    sg_acc = c_m[1, 1] / (c_m[1, 1] + c_m[1, 0])
    non_sg_acc = c_m[0, 0] / (c_m[0, 0] + c_m[0, 1])
    if print_results:
        print("Accuracy: ", acc)
        print("SG Accuracy: ", sg_acc)
        print("Non SG Accuracy: ", non_sg_acc)
        print("Confusion Matrix: ")
        print(c_m)
    return acc, sg_acc, non_sg_acc, c_m


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


def get_obs_to_feats_fn(obs_preprocessor_fn, model_path: str):
    model = PPO.load(model_path)

    def obs_to_feats(obss):
        preproc_obss = obs_preprocessor_fn(obss)
        tensors = obs_as_tensor(preproc_obss, model.device)
        return obs_to_feats_from_model(model, tensors)

    return obs_to_feats


def train_classifier_from_model_path(
    data_manager: datasources.DataSource, model_path: str, test_size: float = 0.0001
) -> dict[str, tuple[BaseStats, BaseStats, dict]]:
    # TODO: stop using this !!    Instead chain
    # `get_obs_to_feats_fn` and `train_classifier_for_extractor`.
    model = PPO.load(model_path)

    def obs_to_feats(obss):
        preproc_obss = data_manager.obs_preprocessor(obss)
        tensors = obs_as_tensor(preproc_obss, model.device)
        return obs_to_feats_from_model(model, tensors)

    return train_classifier_for_extractor(
        data_manager, obs_to_feats, test_size=test_size
    )


def evaluate_extractor(data_manager, clf, obs_to_feats_fn) -> BaseStats:
    obss, images, labels = data_manager.get_data()
    feats = obs_to_feats_fn(obss)
    acc, sg_acc, non_sg_acc, c_m = evaluate(clf, feats, labels, print_results=True)
    return BaseStats(acc=acc, sg_acc=sg_acc, non_sg_acc=non_sg_acc, conf_mat=c_m)


def train_classifier_for_extractor(
    data_source: datasources.DataSource,
    obs_to_feats: Callable[[Iterable], list],
    random_state=None,  # For the train test split
    test_size: float = 0.0,
) -> dict[str, tuple[BaseStats, BaseStats, dict]]:
    """Process a single model."""
    obss, images, labels = data_source.get_data()
    feats = obs_to_feats(obss)
    feat_dim = feats[0].shape[-1]
    classifiers = {
        "linear": sg_detection.LinearClassifier(input_size=feat_dim),
        "nonlinear": sg_detection.NonLinearClassifier(
            input_size=feat_dim, hidden_size=64
        ),
    }

    X = _to_tensor(feats)
    y = torch.tensor(labels).float()
    del feats, labels  # Not to accidentally use them.
    if test_size == 0:
        feats_train = X
        labels_train = y
        feats_test = None
        labels_test = None
    else:
        feats_train, feats_test, labels_train, labels_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    results = {}
    for clf_name, clf in classifiers.items():
        unused_best_acc = train_classifier(
            clf, feats_train, labels_train, disable_tqdm=True
        )
        acc, sg_acc, non_sg_acc, conf_mat = evaluate(clf, feats_train, labels_train)
        details = {
            "classifier": clf,
            "obs_to_feats": obs_to_feats,
        }
        train_stats = BaseStats(
            acc=float(acc),
            sg_acc=float(sg_acc),
            non_sg_acc=float(non_sg_acc),
            conf_mat=conf_mat,
        )
        if feats_test is not None:
            acc, sg_acc, non_sg_acc, conf_mat = evaluate(clf, feats_test, labels_test)
            test_stats = BaseStats(
                acc=float(acc),
                sg_acc=float(sg_acc),
                non_sg_acc=float(non_sg_acc),
                conf_mat=conf_mat,
            )
        else:
            test_stats = None
        results[clf_name] = (train_stats, test_stats, details)
    return results
