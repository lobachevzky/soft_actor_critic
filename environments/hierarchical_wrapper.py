import itertools
from collections.__init__ import namedtuple

import numpy as np
from gym import spaces

from environments.hindsight_wrapper import (FrozenLakeHindsightWrapper,
                                            HindsightWrapper)
from sac.utils import vectorize


class HierarchicalWrapper(HindsightWrapper):
    pass


class FrozenLakeHierarchicalWrapper(HierarchicalWrapper, FrozenLakeHindsightWrapper):
    def __init__(self, env):
        super().__init__(env)
        fl = self.frozen_lake_env
        self._step = fl.s, fl.default_reward, False, {}
        obs = super().reset()

        # DEBUG {{
        # self.observation_space = env.observation_space
        self.observation_space = Hierarchical(
            # DEBUG {{
            boss=env.observation_space,
            # boss=spaces.Box(low=-np.inf, high=np.inf, shape=(
            #     np.shape(vectorize([obs.achieved_goal, obs.desired_goal])))),
            # }}
            worker=spaces.Box(low=-np.inf, high=np.inf, shape=(
                np.shape(vectorize([obs.observation, obs.desired_goal]))))
        )
        # }}

        # DEBUG {{
        # self.action_space = env.action_space
        self.action_space = Hierarchical(

            #     # DEBUG {{
            boss=spaces.Discrete(5),
            #     # boss=spaces.Discrete(2 * (fl.nrow + fl.ncol)),
            #     # }}
            worker=spaces.Discrete(5)
        )
        # }}

    def step(self, action):
        if action != 4:
            self._step = super().step(action)
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
            [0, -1],  # left
            [1, 0],  # down
            [0, 1],  # right
            [-1, 0],  # up
            [0, 0],  # freeze
        ])[goal]
        # return direction / np.linalg.norm(direction)
        # }}


Hierarchical = namedtuple('Hierarchical', 'boss worker')
HierarchicalAgents = namedtuple('HierarchicalAgents', 'boss worker initial_state')
