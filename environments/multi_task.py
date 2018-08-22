import itertools
from collections import namedtuple

import numpy as np
from gym import spaces

from environments.mujoco import distance_between
from environments.pick_and_place import PickAndPlaceEnv

Observation = namedtuple('Obs', 'observation goal')


class MultiTaskEnv(PickAndPlaceEnv):
    def __init__(self,
                 geofence: float,
                 fixed_block=None,
                 fixed_goal=None,
                 goal_x=(-.11, .09),
                 goal_y=(-.19, .2),
                 **kwargs):
        self.fixed_block = fixed_block
        self.fixed_goal = fixed_goal
        self.geofence = geofence
        self.goal_space = spaces.Box(*map(np.array, zip(goal_x, goal_y)))
        self.goal = self.goal_space.sample() if fixed_goal is None else fixed_goal
        super().__init__(fixed_block=False, **kwargs)
        # low=np.array([-.14, -.22, .40]), high=np.array([.11, .22, .63]))
        # goal_size = np.array([.0317, .0635, .0234]) * geofence
        intervals = [2, 3, 1]
        x, y = [
            np.linspace(l, h, n)
            for l, h, n in zip(self.goal_space.low, self.goal_space.high, intervals)
        ]
        goal_corners = np.array(list(itertools.product(x, y)))
        self.labels = {tuple(g) + (.41,): '.' for g in goal_corners}

    def _is_successful(self):
        return distance_between(self.goal, self.block_pos()[:2]) < self.geofence

    def _get_obs(self):
        return Observation(observation=super()._get_obs(), goal=self.goal)

    def _reset_qpos(self):
        block_joint = self.sim.get_jnt_qposadr('block1joint')
        if self.fixed_block is None:
            self.init_qpos[[
                block_joint + 0, block_joint + 1, block_joint + 3, block_joint + 6
            ]] = np.random.uniform(
                low=list(self.goal_space.low) + [0, -1],
                high=list(self.goal_space.high) + [1, 1])
        else:
            self.init_qpos[[block_joint + 0, block_joint + 1,
                            block_joint + 2]] = self.fixed_block
        return self.init_qpos

    def reset(self):
        if self.fixed_goal is None:
            self.goal = self.goal_space.sample()
            self.init_qpos[self.sim.get_jnt_qposadr('goal_x')] = self.goal[0]
            self.init_qpos[self.sim.get_jnt_qposadr('goal_y')] = self.goal[1]
        return super().reset()

    def render(self, labels=None, **kwargs):
        if labels is None:
            labels = dict()
        # z = (.4,)
        # labels[tuple(self.goal_space.low) + z] = '['
        # labels[tuple(self.goal_space.high) + z] = ']'
        # labels[tuple(self.goal) + z] = '|'
        return super().render(labels=labels, **kwargs)
