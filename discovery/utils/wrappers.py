from minigrid.wrappers import ImgObsWrapper

class ImgWithHallwayObsWrapper(ImgObsWrapper):
    def observation(self, obs):
        return {
                'image': obs['image'],
                'hallway': obs['at_hallway']
                }
