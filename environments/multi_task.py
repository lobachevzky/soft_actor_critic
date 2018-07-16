from pprint import pprint

import numpy as np
from gym import spaces
from mujoco import ObjType

from environments.pick_and_place import Goal, PickAndPlaceEnv
import itertools


class MultiTaskEnv(PickAndPlaceEnv):
    def __init__(self, randomize_pose=False, goal_scale: float = .1, **kwargs):
        self.randomize_pose = randomize_pose
        self._goal = None
        super().__init__(fixed_block=False,
                         **kwargs)
        self.goal_space = spaces.Box(
            low=np.array([-.14, -.2240]), high=np.array([.11, .2241]))
        self.goal_size = np.array([.0317, .0635, .0234]) * goal_scale
        # low=np.array([-.14, -.22, .40]), high=np.array([.11, .22, .63]))
        x, y = [np.arange(l, h, s) for l, h, s in
                zip(self.goal_space.low, self.goal_space.high, self.goal_size)]
        z = np.ones_like(x) * .40
        self.goal_corners = np.array(list(itertools.product(x, y, z)))
        self.labels = {tuple(g): '.' for g in self.goal_corners}

    def _set_new_goal(self):
        goal_corner = self.goal_corners[np.random.randint(len(self.goal_corners))]
        self._goal = goal_corner + self.goal_size / 2

    def set_goal(self, goal):
        self._goal = np.array(goal)

    def at_goal(self):
        assert isinstance(self.goal().block, np.ndarray)
        assert isinstance(self.goal_size, np.ndarray)
        block_pos = self.block_pos()
        return np.all((self.goal().block - self.goal_size / 2 <= block_pos) *
                      (self.goal().block + self.goal_size / 2 >= block_pos))

    def goal(self):
        return Goal(gripper=self._goal, block=self._goal)

    def reset_qpos(self):
        if self.randomize_pose:
            for joint in ['slide_x',
                          'slide_y',
                          'arm_lift_joint',
                          'arm_flex_joint',
                          'wrist_roll_joint',
                          'hand_l_proximal_joint']:
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

    def render(self, labels=None, **kwargs):
        if labels is None:
            labels = self.labels
        labels[tuple(self._goal)] = 'x'
        return super().render(labels=labels, **kwargs)
