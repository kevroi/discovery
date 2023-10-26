from __future__ import annotations
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from abc import abstractmethod

class TwoRoomEnv(MiniGridEnv):
    def __init__(
        self,
        # size=8,
        width=15,
        height=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            # grid_size=size,
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
        for i in range(0, 3):
            self.grid.set(7, i, Wall())

        for i in range(4, height):
            self.grid.set(7, i, Wall())
        
        # Place the door and key
        # self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        # self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 5, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "two-room"

def main():
    env = TwoRoomEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()