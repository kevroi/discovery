import os
from stable_baselines3.common.callbacks import BaseCallback


class SnapshotCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir

    def _init_callback(self) -> None:
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.save_path = os.path.join(
                self.log_dir, f"model_snapshot_{self.num_timesteps}_steps"
            )  # self.num_timesteps = n_envs * n times env.step() was called
            print(f"Saving new model to {self.save_path}")
            self.model.save(self.save_path)

        return True

    def _on_training_end(self) -> None:
        self.save_path = os.path.join(self.log_dir, f"model_snapshot_after_training")
        print(f"Saving final model to {self.save_path}")
        self.model.save(self.save_path)
