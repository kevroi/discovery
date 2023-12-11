from __future__ import annotations

import math
import operator
from functools import reduce
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.core import ActionWrapper, ObservationWrapper, ObsType, Wrapper

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.core.world_object import Goal

class RandomEnv(Wrapper):
    def __init__(self, env):
        """A wrapper that randomly picks a new env at the end of each episode.

        Args:
            env: The environment to apply the wrapper
            seeds: A list of seed to be applied to the env
            seed_idx: Index of the initial seed in seeds
        """
        super().__init__(env)

    def reset(self) -> tuple[ObsType, dict[str, Any]]:
        # randomly pick new environment
        
        # pass this to the agent?
        return self.env.reset()