import gymnasium as gym
from gymnasium import spaces
import numpy as np


from gymnasium.envs.registration import register


register(
    id="discovery/Climbing-v0",
    entry_point="environments.climbing:ClimbingEnv",
)

DEFAULT_UP_ACTION = 0
DEFAULT_ANCHOR_ACTION = 1


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

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        render_mode: str = None,
        height: int = 8,
        anchor_interval: int = 4,
        randomized_actions: bool = False,
    ):
        """Construct the environment."""
        self._height = height
        self._anchor_interval = anchor_interval
        self._randomized_actions = randomized_actions
        # If we randomize actions, we may switch the meanings of the two actions.
        self._random_action_sequence = None  # Will be set in `reset`.
        self._agent_location = 0
        self._last_anchor = 0
        self._anchor_locations = np.arange(0, self._height, anchor_interval)[1:]

        # The observations are dictionaries with the agent's location and a bit
        # indicating whether the agent is by an anchor.
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

    def _at_anchor(self):
        return self._agent_location in self._anchor_locations

    def _get_obs(self):
        return {
            "agent_loc": self._agent_location,
            "at_anchor": self._at_anchor(),
            "last_anchor_loc": self._last_anchor,
        }

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

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
        if self._randomized_actions:
            action = self._random_action_sequence[action]

        # Now 0 means move up, and 1 means anchor in.
        if action == DEFAULT_UP_ACTION:
            self._agent_location += 1
        elif action == DEFAULT_ANCHOR_ACTION:
            if self._at_anchor():
                self._last_anchor = self._agent_location
            else:
                self._agent_location = self._last_anchor

        # An episode is done iff the agent has reached the target
        terminated = self._agent_location == self._height - 1
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        return self._render_ansi()

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
