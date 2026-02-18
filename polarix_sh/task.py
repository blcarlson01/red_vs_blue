# polarix_sh/task.py

from polarix import PolarixTask

from polarix_sh.env import PolarixRedvsBlueEnv


class RedvsBluePolarixTask(PolarixTask):
    """
    Polarix task wrapper.
    """

    def __init__(self, config):
        super().__init__()
        self.env = PolarixRedvsBlueEnv(**config)

    def initial_state(self):
        return self.env.reset()

    def transition(self, state, actions):
        obs, reward, done, info = self.env.step(actions)
        return obs, reward, done, info