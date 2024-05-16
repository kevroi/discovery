import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np


from gymnasium.envs.registration import register


register(
    id="discovery/Climbing-v0",
    entry_point="discovery.environments.climbing:ClimbingEnv",
)

DEFAULT_UP_ACTION = 0
DEFAULT_ANCHOR_ACTION = 1

_TILE_PIXS = 8
_AGENT_SIZE_PROP = 0.5
colors = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
}


class ClimbingEnv(gym.Env):
    """An environment with a clear and strong subgoal structure.

    The agent has to climb a mountain, but any wrong move will make it fall down.
    There are anchors on the way up, which the agent can use to save its progress.
    To anchor in, the agent must execute the correct anchoring action when it is
    by the anchor. When the agent falls, it will be reset to the last anchor it
    anchored in --- or to the start if it has not anchored in yet.

    The anchoring locations are observable to the agent, but the agent does not
    a priori know which action is the correct anchoring action, or what anchoring
    even does. The agent must learn this from experience.

    The agent receives a reward of 1 when it reaches the top of the mountain.

    In any state, the agent has two actions. The first action is to move up, and
    the second action is to anchor in.
    TODO: should we have one more action?
    """

    metadata = {"render_modes": ["ansi", "rgb_array"]}

    def __init__(
        self,
        render_mode: str = None,
        height: int = 8,
        anchor_interval: int = 4,
        randomized_actions: bool = False,
        max_episode_length: int = 100,
        include_rgb_obs: bool = False,
    ):
        """Construct the environment."""
        super().__init__()
        self._height = height
        self._anchor_interval = anchor_interval
        self._randomized_actions = randomized_actions
        # If we randomize actions, we may switch the meanings of the two actions.
        self._random_action_sequence = None  # Will be set in `reset`.
        self._agent_location = 0
        self._last_anchor = 0
        self._anchor_locations = np.arange(0, self._height, anchor_interval)[1:]
        self._include_rgb_obs = include_rgb_obs

        # The observations are dictionaries with the agent's location and a bit
        # indicating whether the agent is by an anchor.
        if self._include_rgb_obs:
            self.observation_space = spaces.Dict(
                {
                    "agent_loc": spaces.Discrete(self._height),
                    "at_anchor": spaces.Discrete(2),
                    "last_anchor_loc": spaces.Discrete(self._height),
                    "image": spaces.Box(
                        low=0,
                        high=255,
                        shape=(_TILE_PIXS * self._height, _TILE_PIXS, 3),
                        dtype=np.uint8,
                    ),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent_loc": spaces.Discrete(self._height),
                    "at_anchor": spaces.Discrete(2),
                    "last_anchor_loc": spaces.Discrete(self._height),
                }
            )
        self.action_space = spaces.Discrete(
            2
        )  # 0: move up, 1: anchor in --- unless randomized.

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._max_episode_length = max_episode_length
        self._current_step = 0

    def _at_anchor(self):
        return self._agent_location in self._anchor_locations

    def _get_obs(self):
        obs = {
            "agent_loc": self._agent_location,
            "at_anchor": self._at_anchor(),
            "last_anchor_loc": self._last_anchor,
        }
        if self._include_rgb_obs:
            obs["image"] = self._render_rgb_frame()
        return obs

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)
        self._current_step = 0

        if self._random_action_sequence is None and self._randomized_actions:
            # This way we randomize the actions only the first time we reset,
            # and not on every episode.
            self._random_action_sequence = self.np_random.permutation(2)

        # Choose the agent's location uniformly at random
        self._agent_location = 0
        self._last_anchor = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self._current_step += 1
        if self._randomized_actions:
            action = self._random_action_sequence[action]

        # Now 0 means move up, and 1 means anchor in.
        if action == DEFAULT_UP_ACTION:
            self._agent_location += 1
        elif action == DEFAULT_ANCHOR_ACTION:
            if self._at_anchor():
                self._last_anchor = self._agent_location
                self._agent_location += 1
            else:
                self._agent_location = self._last_anchor

        # An episode is done iff the agent has reached the target
        terminated = self._agent_location == self._height - 1
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self._current_step >= self._max_episode_length:
            truncated = True
        else:
            truncated = False

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb_frame()
        elif self.render_mode == "ansi":
            return self._render_ansi()

    def _render_rgb_frame(self):
        map = np.zeros((self._height * _TILE_PIXS, _TILE_PIXS, 3), dtype=np.uint8)
        for anchor in self._anchor_locations:
            c = colors["gray"] if anchor != self._last_anchor else colors["white"]
            map[anchor * _TILE_PIXS : (anchor + 1) * _TILE_PIXS, 0:_TILE_PIXS] = c
        a_loc_diff = math.floor((0.5 - 0.5 * _AGENT_SIZE_PROP) * _TILE_PIXS)
        start_idx = self._agent_location * _TILE_PIXS + a_loc_diff
        end_idx = (self._agent_location + 1) * _TILE_PIXS - a_loc_diff
        # math.floor(
        #     ( + 0.5 - 0.5 * _AGENT_SIZE_PROP) * _TILE_PIXS
        # )
        # end_idx = math.ceil(
        #     (self._agent_location + 0.5 + 0.5 * _AGENT_SIZE_PROP) * _TILE_PIXS
        # )
        map[start_idx:end_idx, a_loc_diff : _TILE_PIXS - a_loc_diff] = colors["blue"]
        return np.flipud(map)

    def _render_ansi(self):
        map = np.tile(".", [8, 2])
        # map = np.repeat(".", [self._height, 2])
        # map = np.zeros((self._height, 2), dtype=np.character)
        map[self._agent_location, 0] = "A"
        for anchor in self._anchor_locations:
            map[anchor, 1] = "-"
        map[self._last_anchor, 1] = "+"
        map[self._agent_location, 0] = "A"
        return np.flipud(map)

    @property
    def anchor_locations(self):
        return self._anchor_locations.copy()

    @property
    def height(self):
        return self._height

    def close(self):
        pass
