import numpy as np
from gym import spaces
from gym.envs.toy_text import FrozenLakeEnv as gym_env
from gym.envs.toy_text.frozen_lake import MAPS

MAPS["3x3"] = [
    "SFF",
    "FHF",
    "FFG",
]

MAPS["3x4"] = [
    "SFFF",
    "FHFH",
    "HFFG"
]


class FrozenLakeEnv(gym_env):
    def __init__(self, *args, **kwargs):
        self.n_states = None
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(
            low=np.zeros(self.n_states),
            high=np.ones(self.n_states)
        )

    def reset(self):
        return self.one_hotify(super().reset())

    def step(self, a):
        s, r, t, i = super().step(a)
        return self.one_hotify(s), r, t, i

    def one_hotify(self, s):
        if not self.n_states:
            self.n_states = self.observation_space.n
        array = np.zeros(self.n_states)
        array[s] = 1
        return array
