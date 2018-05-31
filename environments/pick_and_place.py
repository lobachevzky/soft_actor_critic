import random
from collections import namedtuple
from contextlib import contextmanager
from copy import deepcopy
from os.path import join

import numpy as np
from gym import spaces

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
                 fixed_block,
                 min_lift_height=.02,
                 geofence=.04,
                 neg_reward=False,
                 discrete=False,
                 cheat_prob=0):
        self._cheated = False
        self._cheat_prob = cheat_prob
        self.grip = 0
        self._fixed_block = fixed_block
        self._goal_block_name = 'block1'
        self._min_lift_height = min_lift_height + geofence
        self.geofence = geofence
        self._discrete = discrete

        super().__init__(
            xml_filepath=join('models', 'pick-and-place', 'discrete.xml'
                              if discrete else 'world.xml'),
            neg_reward=neg_reward,
            steps_per_action=20,
            image_dimensions=None)

        self.init_qpos = deepcopy(self.sim.get_state().qpos)
        self._initial_block_pos = np.copy(self.block_pos())
        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [left_finger_name, left_finger_name.replace('_l_', '_r_')]
        obs_size = sum(map(np.size, self._get_obs()))
        assert obs_size != 0
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_size, ), dtype=np.float32)
        if discrete:
            self.action_space = spaces.Discrete(7)
        else:
            self.action_space = spaces.Box(
                low=np.array([-15, -20, -20]),
                high=np.array([35, 20, 20]),
                dtype=np.float32)
        self._table_height = self.sim.data.get_body_xpos('pan')[2]
        self._rotation_actuators = ["arm_flex_motor"]  # , "wrist_roll_motor"]

    def reset_qpos(self):
        # slide_y = self.sim.model.get_joint_qpos_addr('slide_y')
        # self.init_qpos[slide_y] = np.random.uniform(-0.2, 0.2)
        # arm_joint = self.sim.model.get_joint_qpos_addr('arm_flex_joint')
        # self.init_qpos[arm_joint] = np.random.uniform(-0.960114368248, 0.00101480673663)
        # wrist_joint = self.sim.model.get_joint_qpos_addr('wrist_roll_joint')
        # self.init_qpos[wrist_joint] = np.random.uniform(-1.5744836894, 1.57448370861)
        # l_hand_joint = self.sim.model.get_joint_qpos_addr('hand_l_proximal_joint')
        # self.init_qpos[l_hand_joint] = np.random.uniform(-0.00842414027907, 0.357219407462)
        # r_hand_joint = self.sim.model.get_joint_qpos_addr('hand_r_proximal_joint')
        # self.init_qpos[r_hand_joint] = self.init_qpos[l_hand_joint]

        if not self._fixed_block:
            block_joint, _ = self.sim.model.get_joint_qpos_addr('block1joint')
            self.init_qpos[block_joint + 1] = np.random.uniform(-0.2, 0.2)
            self.init_qpos[block_joint + 3] = np.random.uniform(-np.pi, np.pi)
            self.init_qpos[block_joint + 6] = np.random.uniform(-np.pi, np.pi)
        if np.random.uniform(0, 1) < self._cheat_prob:
            self._cheated = True
            self.init_qpos = np.array(random.choice(CHEAT_STARTS))
        else:
            self._cheated = False
        return self.init_qpos

    def _get_obs(self):
        return np.copy(self.sim.data.qpos)

    def block_pos(self):
        return self.sim.data.get_body_xpos(self._goal_block_name)

    def gripper_pos(self):
        finger1, finger2 = [
            self.sim.data.get_body_xpos(name) for name in self._finger_names
        ]
        return (finger1 + finger2) / 2.

    def goal(self):
        goal_pos = self._initial_block_pos + \
            np.array([0, 0, self._min_lift_height])
        return Goal(gripper=goal_pos, block=goal_pos)

    def goal_3d(self):
        return self.goal()[0]

    def _set_new_goal(self):
        pass

    def compute_reward(self):
        if self.block_pos()[2] > self._initial_block_pos[2] + self._min_lift_height:
            return 1
        elif self._neg_reward:
            return -.0001
        else:
            return 0

    def step(self, action):
        if self._discrete:
            a = np.zeros(4)
            if action > 0:
                action -= 1
                joint = action // 2
                assert 0 <= joint <= 2
                direction = (-1)**(action % 2)
                joint_scale = [.2, .05, .5]
                a[2] = self.grip
                a[joint] = direction * joint_scale[joint]
                self.grip = a[2]
            action = a
        action = np.clip(action, self.action_space.low, self.action_space.high)

        mirrored = 'hand_l_proximal_motor'
        mirroring = 'hand_r_proximal_motor'

        # insert mirrored values at the appropriate indexes
        mirrored_index, mirroring_index = [
            self.sim.model.actuator_name2id(n) for n in [mirrored, mirroring]
        ]
        # necessary because np.insert can't append multiple values to end:
        if self._discrete:
            action[mirroring_index] = action[mirrored_index]
        else:
            mirroring_index = np.minimum(mirroring_index, self.action_space.shape)
            action = np.insert(action, mirroring_index, action[mirrored_index])
        s, r, t, i = super().step(action)
        if not self._cheated:
            i['log count'] = {'successes': float(r > 0)}
        return s, r, t, i
