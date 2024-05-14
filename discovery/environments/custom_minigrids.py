from __future__ import annotations
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from gymnasium import spaces
from abc import abstractmethod
import random

from minigrid.core.constants import (
    IDX_TO_OBJECT,
)

from stable_baselines3.common.vec_env.util import obs_space_info


class TwoRoomEnv(MiniGridEnv):
    def __init__(
        self,
        width=15,
        height=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        hallway_pos=(3, 7),
        random_hallway=False,  # if True, hallway position is randomly generated (we uniformly sample from a distribution of TwoRoomEnvs)
        max_steps: int = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.hallway_pos = hallway_pos
        self.random_hallway = random_hallway

        mission_space = MissionSpace(mission_func=self._gen_mission)
        # Randomly place hallway
        if self.random_hallway:
            self.hallway_pos = (random.randint(1, height - 2), self.hallway_pos[1])
            self.num_variants = height - 2  # number of different TwoRoomEnvs
            self.variant_idx = self.hallway_pos[0]  # index of the current TwoRoomEnv

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=1000,
            **kwargs,
        )

        self.observation_space["at_hallway"] = spaces.Discrete(2)

    @staticmethod
    def _gen_mission():
        return "grand mission"

    # MiniGridEnv._gen_grid
    @abstractmethod
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate verical separation wall
        for i in range(0, self.hallway_pos[0]):
            self.grid.set(self.hallway_pos[1], i, Wall())

        for i in range(self.hallway_pos[0] + 1, height):
            self.grid.set(self.hallway_pos[1], i, Wall())

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 5, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "two-room"

    def gen_obs(self):
        obs = super().gen_obs()
        obs["at_hallway"] = self.agent_pos == self.hallway_pos
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.gen_obs()

        return observation, {}


class FourRoomEnv(MiniGridEnv):
    """
    Asymmetric 4-room environment.
    """

    def __init__(
        self,
        width=13,
        height=13,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        hallway_pos=[(3, 6), (6, 2), (7, 9), (10, 6)],
        max_steps: int = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.hallway_pos = hallway_pos

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=1000,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    # MiniGridEnv._gen_grid
    @abstractmethod
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate verical separation wall
        for i in range(1, 3):
            self.grid.set(6, i, Wall())

        for i in range(4, 10):
            self.grid.set(6, i, Wall())

        for i in range(11, 12):
            self.grid.set(6, i, Wall())

        # Generate horizontal separation wall
        for i in range(1, 2):
            self.grid.set(i, 6, Wall())

        for i in range(3, 6):
            self.grid.set(i, 6, Wall())

        for i in range(7, 9):
            self.grid.set(i, 7, Wall())

        for i in range(10, 12):
            self.grid.set(i, 7, Wall())

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "four-room"


class FourRoomChainEnv(MiniGridEnv):
    def __init__(
        self,
        width=29,
        height=8,
        random_starts=True,
        random_goal=False,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        hallway_pos=[(3, 7), (5, 14), (1, 21), (4, 28), (3, 35)],
        # goal_set = [(1,1), (1,6), (27, 1), (27, 6)], # four corners - only used if random_goal is True
        goal_set=[(24, 6), (4, 6)],
        max_steps: int = None,
        **kwargs,
    ):

        if random_starts:
            self.agent_start_pos = None
        else:
            self.agent_start_pos = agent_start_pos
            self.agent_start_dir = agent_start_dir
        self.hallway_pos = hallway_pos
        self.random_goal = random_goal
        self.goal_set = goal_set
        random.seed(0)  # seed for goal positions

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=2000,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    # MiniGridEnv._gen_grid
    @abstractmethod
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate verical separation wall
        for i in range(0, self.hallway_pos[0][0]):
            self.grid.set(self.hallway_pos[0][1], i, Wall())
        for i in range(self.hallway_pos[0][0] + 1, height):
            self.grid.set(self.hallway_pos[0][1], i, Wall())

        for i in range(0, self.hallway_pos[1][0]):
            self.grid.set(self.hallway_pos[1][1], i, Wall())
        for i in range(self.hallway_pos[1][0] + 1, height):
            self.grid.set(self.hallway_pos[1][1], i, Wall())

        for i in range(0, self.hallway_pos[2][0]):
            self.grid.set(self.hallway_pos[2][1], i, Wall())
        for i in range(self.hallway_pos[2][0] + 1, height):
            self.grid.set(self.hallway_pos[2][1], i, Wall())

        # Place a goal square in the bottom-right corner
        if self.random_goal:
            goal_pos = random.choice(self.goal_set)
            self.put_obj(Goal(), goal_pos[0], goal_pos[1])
        else:
            self.put_obj(Goal(), width - 5, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "4-room-chain"


class SixRoomChainEnv(MiniGridEnv):
    def __init__(
        self,
        width=43,
        height=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        hallway_pos=[(3, 7), (5, 14), (1, 21), (4, 28), (3, 35)],
        max_steps: int = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.hallway_pos = hallway_pos

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=2500,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    # MiniGridEnv._gen_grid
    @abstractmethod
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate verical separation wall
        for i in range(0, self.hallway_pos[0][0]):
            self.grid.set(self.hallway_pos[0][1], i, Wall())
        for i in range(self.hallway_pos[0][0] + 1, height):
            self.grid.set(self.hallway_pos[0][1], i, Wall())

        for i in range(0, self.hallway_pos[1][0]):
            self.grid.set(self.hallway_pos[1][1], i, Wall())
        for i in range(self.hallway_pos[1][0] + 1, height):
            self.grid.set(self.hallway_pos[1][1], i, Wall())

        for i in range(0, self.hallway_pos[2][0]):
            self.grid.set(self.hallway_pos[2][1], i, Wall())
        for i in range(self.hallway_pos[2][0] + 1, height):
            self.grid.set(self.hallway_pos[2][1], i, Wall())

        for i in range(0, self.hallway_pos[3][0]):
            self.grid.set(self.hallway_pos[3][1], i, Wall())
        for i in range(self.hallway_pos[3][0] + 1, height):
            self.grid.set(self.hallway_pos[3][1], i, Wall())

        for i in range(0, self.hallway_pos[4][0]):
            self.grid.set(self.hallway_pos[4][1], i, Wall())
        for i in range(self.hallway_pos[4][0] + 1, height):
            self.grid.set(self.hallway_pos[4][1], i, Wall())

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "6-room-chain"


def main():
    # gym.register(id="FourRoomChainEnv", entry_point=FourRoomChainEnv)
    # # env = FourRoomChainEnv(render_mode="human", random_goal=True)
    # def make_env(config):
    #     env = gym.make(config['env_name'], render_mode='rgb_array')
    #     env = RGBImgObsWrapper(env) # FullyObsWrapper runs faster locally, but uses ints instead of 256-bit RGB
    #     env = ImgObsWrapper(env)
    #     env = Monitor(env)
    #     # wrappers = [
    #     #     lambda env: RGBImgObsWrapper(env),
    #     #     lambda env: ImgObsWrapper(env),
    #     #     # lambda env: Monitor(env),
    #     # ]
    #     # env = gym.make(config['env_name'], render_mode='rgb_array')
    #     # for wrapper in wrappers:
    #     #     env = wrapper(env)
    #     return env
    # conf = dict(env_name="FourRoomChainEnv",
    #             n_envs=1,
    #             seed=42,
    #             )
    # # env = DummyVecEnv([lambda: make_env(config=conf)]*conf['n_envs'])
    # env = make_env(config=conf)
    # # from stable_baselines3.common.vec_env.patch_gym import _patch_env
    # # envs = [fn() for fn in [lambda: make_env(config=conf)]*5]
    # obs, _ = env.reset()
    # # print(obs.shape)
    # print(env.observation_space)

    # # # enable manual control for testing
    # # manual_control = ManualControl(env, seed=42)
    # # manual_control.start()

    env = TwoRoomEnv(render_mode="human", random_hallway=True)

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()
