from environments.environment import ToyEnvironment
from eigenoptions.options import Options

# Environment we wish to find eigenvectors for
opt_env = ToyEnvironment('4room')
opt = Options(opt_env, alpha=0.1, epsilon=1.0, discount=0.9)
opt.display_eigenvector(env=opt_env)