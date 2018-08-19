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
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile

    def goal_to_boss_action_space(self, goal: np.array):
        i, j = goal
        n = self.action_space.boss.n
        action = np.zeros(n)
        action[i * int(np.sqrt(n)) + j] = 1
        return action

    def boss_action_to_goal_space(self, action: np.array):
        action = np.argmax(action)
        n = np.sqrt(self.action_space.boss.n)
        return np.array([action // n, action % n])


Hierarchical = namedtuple('Hierarchical', 'boss worker')
HierarchicalAgents = namedtuple('HierarchicalAgents', 'boss worker initial_state')
