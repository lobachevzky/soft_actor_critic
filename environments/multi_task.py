from pathlib import Path

import numpy as np
from gym import spaces
from mujoco import ObjType

from environments.pick_and_place import Goal, PickAndPlaceEnv


class MultiTaskEnv(PickAndPlaceEnv):
    def __init__(self, steps_per_action, geofence, min_lift_height, render_freq, fixed_pose=True):
        self.fixed_pose = fixed_pose
        self._goal = None
        super().__init__(
            fixed_block=False,
            steps_per_action=steps_per_action,
            geofence=geofence,
            min_lift_height=min_lift_height,
            xml_filepath=Path('models', '6dof', 'world.xml'),
            render_freq=render_freq)
        self.goal_space = spaces.Box(
            low=np.array([-.14, -.22, .45]), high=np.array([.11, .22, .63]))
        self.goals = [
            np.linspace(start, stop, num) for start, stop, num in zip(
                self.goal_space.low, self.goal_space.high, [1, 1, 1])
        ]

    def _set_new_goal(self):
        self._goal = np.array([np.random.choice(x) for x in self.goals])

    def set_goal(self, goal):
        self._goal = np.array(goal)

    def goal(self):
        return Goal(gripper=self._goal, block=self._goal)

    def goal_3d(self):
        return self._goal

    def reset_qpos(self):
        if not self.fixed_pose:
            for joint in ['slide_x',
                          'slide_y',
                          'arm_lift_joint',
                          'arm_flex_joint',
                          'wrist_roll_joint',
                          'hand_l_proximal_joint']:
                qpos_idx = self.sim.jnt_qposadr(joint)
                jnt_range_idx = self.sim.name2id(ObjType.JOINT, joint)
                self.init_qpos[qpos_idx] = np.random.uniform(*self.sim.jnt_range[jnt_range_idx])

        r = self.sim.jnt_qposadr('hand_r_proximal_joint')
        l = self.sim.jnt_qposadr('hand_l_proximal_joint')
        self.init_qpos[r] = self.init_qpos[l]

        block_joint = self.sim.jnt_qposadr('block1joint')
        self.init_qpos[[
            block_joint + 0, block_joint + 1, block_joint + 3, block_joint + 6
        ]] = np.random.uniform(
            low=list(self.goal_space.low)[:2] + [0, -1],
            high=list(self.goal_space.high)[:2] + [1, 1])
        return self.init_qpos
