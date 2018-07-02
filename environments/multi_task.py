import itertools
from collections import namedtuple

import numpy as np
from gym import spaces

from environments.pick_and_place import PickAndPlaceEnv
from mujoco import ObjType
from sac.utils import vectorize

Observation = namedtuple('Obs', 'observation goal')


class MultiTaskEnv(PickAndPlaceEnv):
    def __init__(self, randomize_pose=False, goal_scale: float = .1, **kwargs):
        self.randomize_pose = randomize_pose
        self._goal = None
        super().__init__(fixed_block=False, **kwargs)
        self.goal_space = spaces.Box(
            low=np.array([-.14, -.2240]), high=np.array([.11, .2241]))
        # self.goal_size = np.array([.0317, .0635, .0234]) * goal_scale
        self.goal_size = np.array([.0317, .0635]) * goal_scale
        x, y = [
            np.arange(l, h, s)
            for l, h, s in zip(self.goal_space.low, self.goal_space.high, self.goal_size)
        ]
        self.goal_corners = np.array(list(itertools.product(x, y)))
        self.labels = {tuple(g) + (.4, ): '.' for g in self.goal_corners}
        self.observation_space = spaces.Box(
            low=vectorize([self.observation_space.low, self.goal_space.low]),
            high=vectorize([self.observation_space.high, self.goal_space.high]))

    def _is_successful(self):
        assert isinstance(self.goal, np.ndarray)
        assert isinstance(self.goal_size, np.ndarray)
        pos = self.block_pos()[:2]
        return np.all((self.goal - self.goal_size / 2 <= pos) *
                      (self.goal + self.goal_size / 2 >= pos))

    def _get_obs(self):
        return Observation(observation=super()._get_obs(), goal=self.goal)

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, goal):
        self._goal = goal

    def _reset_qpos(self):
        if self.randomize_pose:
            for joint in [
                    'slide_x', 'slide_y', 'arm_lift_joint', 'arm_flex_joint',
                    'wrist_roll_joint', 'hand_l_proximal_joint'
            ]:
                qpos_idx = self.sim.get_jnt_qposadr(joint)
                jnt_range_idx = self.sim.name2id(ObjType.JOINT, joint)
                self.init_qpos[qpos_idx] = np.random.uniform(
                    *self.sim.jnt_range[jnt_range_idx])

        r = self.sim.get_jnt_qposadr('hand_r_proximal_joint')
        l = self.sim.get_jnt_qposadr('hand_l_proximal_joint')
        self.init_qpos[r] = self.init_qpos[l]

        block_joint = self.sim.get_jnt_qposadr('block1joint')
        self.init_qpos[[
            block_joint + 0, block_joint + 1, block_joint + 3, block_joint + 6
        ]] = np.random.uniform(
            low=list(self.goal_space.low)[:2] + [0, -1],
            high=list(self.goal_space.high)[:2] + [1, 1])
        return self.init_qpos

    def reset(self):
        goal_corner = self.goal_corners[np.random.randint(len(self.goal_corners))]
        self.goal = goal_corner + self.goal_size / 2
        return super().reset()

    def render(self, labels=None, **kwargs):
        if labels is None:
            labels = self.labels
        labels[tuple(self._goal)] = 'x'
        return super().render(labels=labels, **kwargs)
