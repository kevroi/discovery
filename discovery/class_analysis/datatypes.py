from enum import Enum
from dataclasses import dataclass

import numpy as np


class ModelType(Enum):
    CNN = 1
    FTA = 2
    RANDOM_PROJ_GAUSS = 3
    RANDOM_PROJ_SPARSE = 4
    OBSERVATION = 5


class EnvName(Enum):
    TwoRooms = 1
    Seaquest = 2


@dataclass(frozen=True, eq=True)
class Setting:
    multitask: bool
    model_type: ModelType
    env_name: EnvName


@dataclass
class Data:
    wandb_ids: list[str]  # For random projection models this is undefined.
    num_runs: int
    lin_accuracies: list[float]
    lin_sg_accuracies: list[float]
    lin_non_sg_accuracies: list[float]
    lin_conf_matrices: list[np.array]
    lin_acc_mean: float
    lin_acc_std_err: float
    lin_sg_acc_mean: float
    lin_sg_acc_std_err: float
    lin_non_sg_acc_mean: float
    lin_non_sg_acc_std_err: float
    nonlin_accuracies: list[float]
    nonlin_sg_accuracies: list[float]
    nonlin_non_sg_accuracies: list[float]
    nonlin_conf_matrices: list[np.array]
    nonlin_acc_mean: float
    nonlin_acc_std_err: float
    nonlin_sg_acc_mean: float
    nonlin_sg_acc_std_err: float
    nonlin_non_sg_acc_mean: float
    nonlin_non_sg_acc_std_err: float
