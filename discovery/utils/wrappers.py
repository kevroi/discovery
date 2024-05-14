from minigrid.wrappers import ImgObsWrapper
from gymnasium import spaces


class ImgWithHallwayObsWrapper(ImgObsWrapper):

    def __init__(self, env):
        """A wrapper that makes image and hallway the only observations.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                k: s
                for k, s in env.observation_space.spaces.items()
                if k in ["image", "at_hallway"]
            }
        )

    def observation(self, obs):
        return {"image": obs["image"], "at_hallway": obs["at_hallway"]}
