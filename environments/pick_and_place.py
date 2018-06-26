import random
from collections import namedtuple, Iterable
from pathlib import Path

import numpy as np
from gym import spaces
from mujoco import ObjType

from environments.mujoco import MujocoEnv

CHEAT_STARTS = [[
    7.450e-05,
    -3.027e-03,
    4.385e-01,
    1.000e+00,
    0,
    0,
    -6.184e-04,
    -1.101e+00,
    0,
    3.573e-01,
    3.574e-01,
], [-0.005, 0.025, 0.447, 0.488, -0.488, -0.512, 0.512, -1.101, 1.575, 0.357, 0.357], [
    4.636e-03, 8.265e-06, 4.385e-01, 7.126e-01, 2.072e-17, -2.088e-17, -7.015e-01,
    -1.101e+00, -1.575e+00, 3.573e-01, 3.574e-01
], [
    5.449e-03, -4.032e-03, 4.385e-01, 3.795e-01, 1.208e-17, -2.549e-17, -9.252e-01,
    -1.101e+00, -7.793e-01, 3.573e-01, 3.574e-01
]]


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array(
        [
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0, x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0, x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
        ],
        dtype=np.float64)


Goal = namedtuple('Goal', 'gripper block')


class PickAndPlaceEnv(MujocoEnv):
    def __init__(self,
                 xml_filepath,
                 steps_per_action,
                 fixed_block=False,
                 min_lift_height=.02,
                 geofence=.04,
                 neg_reward=False,
                 cheat_prob=0,
                 render_freq=0,
                 obs_type=None):
        self._obs_type = obs_type
        self._cheated = False
        self._cheat_prob = cheat_prob
        self.grip = 0
        self._fixed_block = fixed_block
        self._goal_block_name = 'block1'
        self._min_lift_height = min_lift_height
        self.geofence = geofence

        super().__init__(
            xml_filepath=xml_filepath,
            neg_reward=neg_reward,
            steps_per_action=steps_per_action,
            image_dimensions=None,
            render_freq=render_freq,
        )

        self.initial_qpos = np.copy(self.init_qpos)
        self._initial_block_pos = np.copy(self.block_pos())
        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [left_finger_name, left_finger_name.replace('_l_', '_r_')]
        obs_size = sum(map(np.size, self._get_obs()))
        assert obs_size != 0
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.sim.actuator_ctrlrange[:-1, 0],
            high=self.sim.actuator_ctrlrange[:-1, 1],
            dtype=np.float32)
        self._table_height = self.sim.get_body_xpos('pan')[2]
        self._rotation_actuators = ["arm_flex_motor"]  # , "wrist_roll_motor"]
        self.unwrapped = self

    def reset_qpos(self):
        if np.random.uniform(0, 1) < self._cheat_prob:
            self._cheated = True
            self.init_qpos = np.array(random.choice(CHEAT_STARTS))
        else:
            self._cheated = False
            self.init_qpos = self.initial_qpos
        if not self._fixed_block:
            block_joint = self.sim.jnt_qposadr('block1joint')
            # self.init_qpos[block_joint] = np.random.uniform(-.1, .1)
            self.init_qpos[block_joint + 3] = np.random.uniform(0, 1)
            self.init_qpos[block_joint + 6] = np.random.uniform(-1, 1)

        return self.init_qpos

    def _set_new_goal(self):
        pass

    def _get_obs(self):
        def get_qvels(joints):
            base_qvel = []
            for joint in joints:
                try:
                    base_qvel.append(self.sim.get_joint_qvel(joint))
                except RuntimeError:
                    pass
            return np.array(base_qvel)

        if self._obs_type == 'qvel':
            qvel = self.sim.qvel

        elif self._obs_type == 'robot-qvel':
            qvel = get_qvels(['slide_x', 'slide_y',
                              'arm_lift_joint',
                              'arm_flex_joint',
                              'wrist_roll_joint',
                              'hand_l_proximal_joint',
                              'hand_r_proximal_joint'])
        elif self._obs_type == 'base-qvel':
            qvel = get_qvels(['slide_x', 'slide_x'])
        else:
            qvel = []

        return np.concatenate([self.sim.qpos, qvel])

    def block_pos(self):
        return self.sim.get_body_xpos(self._goal_block_name)

    def gripper_pos(self):
        finger1, finger2 = [self.sim.get_body_xpos(name) for name in self._finger_names]
        return (finger1 + finger2) / 2.

    def goal(self):
        goal_pos = self._initial_block_pos + \
                   np.array([0, 0, self._min_lift_height])
        return Goal(gripper=goal_pos, block=goal_pos)

    def goal_3d(self):
        return self.goal()[0]

    def at_goal(self, _):
        return self.block_pos()[2] > self._initial_block_pos[2] + self._min_lift_height

    def compute_terminal(self, goal, obs):
        # return False
        return self.at_goal(goal)

    def compute_reward(self, goal, obs):
        if self.at_goal(goal):
            return 1
        elif self._neg_reward:
            return -.0001
        else:
            return 0

    def step(self, action):
        # action = np.array([1, 1, 0, 0, 0, 0])
        action = np.clip(action, self.action_space.low, self.action_space.high)

        mirrored = 'hand_l_proximal_motor'
        mirroring = 'hand_r_proximal_motor'

        # insert mirrored values at the appropriate indexes
        mirrored_index, mirroring_index = [
            self.sim.name2id(ObjType.ACTUATOR, n) for n in [mirrored, mirroring]
            ]
        # necessary because np.insert can't append multiple values to end:
        mirroring_index = np.minimum(mirroring_index, self.action_space.shape)
        action = np.insert(action, mirroring_index, action[mirrored_index])

        s, r, t, i = super().step(action)
        if not self._cheated:
            i['log count'] = {'successes': float(r > 0)}
        return s, r, t, i


def filter_none(it: Iterable):
    return [x for x in it if x is not None]
