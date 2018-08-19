import itertools
import sys

from collections.__init__ import namedtuple
from typing import Tuple

import numpy as np
from gym import spaces, utils

from environments.hindsight_wrapper import (FrozenLakeHindsightWrapper,
                                            HindsightWrapper, Observation)
from sac.utils import vectorize
from six import StringIO, b


class HierarchicalWrapper(HindsightWrapper):
    def goal_to_boss_action_space(self, goal: np.array):
        raise NotImplemented

    def boss_action_to_goal_space(self, goal: np.array):
        raise NotImplemented


class FrozenLakeHierarchicalWrapper(HierarchicalWrapper, FrozenLakeHindsightWrapper):
    def __init__(self, env):
        super().__init__(env)
        fl = self.frozen_lake_env
        obs = super().reset()
        self._step = obs, fl.default_reward, False, {}

        self.observation_space = Hierarchical(
            boss=spaces.Box(low=-np.inf, high=np.inf, shape=(
                np.shape(vectorize([obs.achieved_goal, obs.desired_goal])))),
            worker=spaces.Box(low=-np.inf, high=np.inf, shape=(
                np.shape(vectorize([obs.observation, obs.desired_goal]))))
        )

        self.action_space = Hierarchical(
            # DEBUG {{
            boss=spaces.Discrete(9),
            # boss=spaces.Discrete(1 + 2 * (fl.nrow + fl.ncol)),
            # }}
            worker=spaces.Discrete(5)
        )

    # DEBUG {{
    def step(self, direction: int):
        s, r, t, i = super().step(direction)
        new_s = Observation(
            observation=s.observation,
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())
        return new_s, r, t, i
    # }}

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        fl = self.frozen_lake_env
        row, col = fl.s // fl.ncol, fl.s % fl.ncol
        desc = fl.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if fl.lastaction is not None:
            print('last action:', fl.lastaction)
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile

    def get_direction(self, goal: int):
        fl = self.frozen_lake_env
        i = itertools.chain(
            [0],
            [-fl.nrow + 1] * (fl.ncol * 2 - 2),
            range(-fl.nrow + 1, fl.nrow - 1),
            [fl.nrow - 1] * (fl.ncol * 2 - 2),
            range(fl.nrow - 1, -fl.nrow + 1, -1),
        )
        j = itertools.chain(
            [0],
            range(-fl.ncol + 1, fl.ncol - 1),
            [fl.ncol - 1] * (fl.nrow * 2 - 2),
            range(fl.ncol - 1, -fl.ncol + 1, -1),
            [-fl.ncol + 1] * (fl.nrow * 2 - 2),
        )

        # i = itertools.chain(
        #     [0],
        #     [-1] * 2,
        #     range(-1, 1),
        #     [1] * 2,
        #     range(1, -1, -1),
        #     )
        #
        # j = itertools.chain(
        #     [0],
        #     range(-1, 1),
        #     [1] * 2,
        #     range(1, -1, -1),
        #     [-1] * 2,
        #     )
        return np.array([(goal // 3) - 1, (goal % 3) - 1])
        # l = list(zip(i, j))
        # direction = np.array(l[goal], dtype=float)
        # if not np.allclose(direction, 0):
        #     direction /= np.linalg.norm(direction)
        # return direction

        # return np.array([
        #     [0, 0],  # freeze
        #     [0, -1],  # left
        #     [1, 0],  # down
        #     [0, 1],  # right
        #     [-1, 0],  # up
        # ])[goal]


Hierarchical = namedtuple('Hierarchical', 'boss worker')
HierarchicalAgents = namedtuple('HierarchicalAgents', 'boss worker initial_state')
