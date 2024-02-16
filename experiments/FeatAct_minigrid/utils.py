import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, FullyObsWrapper
from stable_baselines3.common.monitor import Monitor

# HELPER FUNCTIONS ##
def make_env(config):
    env = gym.make(config['env_name'], render_mode='rgb_array')
    env = RGBImgObsWrapper(env) # FullyObsWrapper runs faster locally, but uses ints instead of 256-bit RGB
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env
#####################