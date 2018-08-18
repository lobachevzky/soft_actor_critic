import itertools
from collections.__init__ import namedtuple

import numpy as np
from gym import spaces

from environments.hindsight_wrapper import (FrozenLakeHindsightWrapper,
                                            HindsightWrapper, Observation)
from sac.utils import vectorize


class HierarchicalWrapper(HindsightWrapper):
    pass


class FrozenLakeHierarchicalWrapper(HierarchicalWrapper, FrozenLakeHindsightWrapper):
    def __init__(self, env):
        super().__init__(env)
        fl = self.frozen_lake_env
        obs = super().reset()
        self._step = obs, fl.default_reward, False, {}

        self.observation_space = Hierarchical(
            # DEBUG {{
            boss=env.observation_space,
            # boss=spaces.Box(low=-np.inf, high=np.inf, shape=(
            #     np.shape(vectorize([obs.achieved_goal, obs.desired_goal])))),
            # }}
            worker=spaces.Box(low=-np.inf, high=np.inf, shape=(
                np.shape(vectorize([obs.observation, obs.desired_goal]))))
        )

        self.action_space = Hierarchical(

            #     # DEBUG {{
            boss=spaces.Discrete(5),
            #     # boss=spaces.Discrete(2 * (fl.nrow + fl.ncol)),
            #     # }}
            worker=spaces.Discrete(5)
        )

    def step(self, action: int):
        if action != 0:
            self._step = super().step(action - 1)
        return self._step

    def get_direction(self, goal: int):
        fl = self.frozen_lake_env
        i = itertools.chain(
            [-fl.nrow] * fl.ncol,
            range(fl.nrow),
            [fl.nrow] * fl.ncol,
            range(fl.nrow),
        )
        j = itertools.chain(
            range(fl.ncol),
            [fl.ncol] * fl.nrow,
            range(fl.ncol),
            [-fl.ncol] * fl.nrow,
        )
        direction = list(zip(i, j))[goal]

        # DEBUG {{
        return np.array([
            [0, 0],  # freeze
            [0, -1],  # left
            [1, 0],  # down
            [0, 1],  # right
            [-1, 0],  # up
        ])[goal]
        # return direction / np.linalg.norm(direction)
        # }}


Hierarchical = namedtuple('Hierarchical', 'boss worker')
HierarchicalAgents = namedtuple('HierarchicalAgents', 'boss worker initial_state')
