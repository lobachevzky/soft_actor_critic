import random

import numpy as np
from gym import spaces

from environments.mujoco import MujocoEnv
from mujoco import ObjType
from sac.utils import vectorize

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


class LiftEnv(MujocoEnv):
    def __init__(self,
                 block_xrange=None,
                 block_yrange=None,
                 fixed_block=False,
                 min_lift_height=.02,
                 cheat_prob=0,
                 **kwargs):
        if block_xrange is None:
            block_xrange = (0, 0)
        if block_yrange is None:
            block_yrange = (0, 0)
        self.block_xrange = block_xrange
        self.block_yrange = block_yrange
        self.grip = 0
        self.min_lift_height = min_lift_height

        self._cheated = False
        self._cheat_prob = cheat_prob
        self._fixed_block = fixed_block
        self._block_name = 'block1'

        super().__init__(**kwargs)

        self.reward_range = 0, 1
        self.initial_qpos = np.copy(self.init_qpos)
        self.initial_block_pos = np.copy(self.block_pos())
        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [left_finger_name, left_finger_name.replace('_l_', '_r_')]
        self.observation_space = self._get_obs_space()
        self.action_space = spaces.Box(
            low=self.sim.actuator_ctrlrange[:-1, 0],
            high=self.sim.actuator_ctrlrange[:-1, 1],
            dtype=np.float32)
        self._table_height = self.sim.get_body_xpos('pan')[2]
        self._rotation_actuators = ["arm_flex_motor"]  # , "wrist_roll_motor"]
        self.unwrapped = self

    def _reset_qpos(self):
        if np.random.uniform(0, 1) < self._cheat_prob:
            self._cheated = True
            self.init_qpos = np.array(random.choice(CHEAT_STARTS))
        else:
            self._cheated = False
            self.init_qpos = self.initial_qpos
        if not self._fixed_block:
            block_joint = self.sim.get_jnt_qposadr('block1joint')
            self.init_qpos[block_joint + 0] = np.random.uniform(*self.block_xrange)
            self.init_qpos[block_joint + 1] = np.random.uniform(*self.block_yrange)
            self.init_qpos[block_joint + 3] = np.random.uniform(0, 1)
            self.init_qpos[block_joint + 6] = np.random.uniform(-1, 1)

        return self.init_qpos

    def _get_obs_space(self):
        qpos_limits = [(-np.inf, np.inf) for _ in self.sim.qpos]
        qvel_limits = [(-np.inf, np.inf) for _ in self._qvel_obs()]
        for joint_id in range(self.sim.njnt):
            if self.sim.get_jnt_type(joint_id) in ['mjJNT_SLIDE', 'mjJNT_HINGE']:
                qposadr = self.sim.get_jnt_qposadr(joint_id)
                qpos_limits[qposadr] = self.sim.jnt_range[joint_id]
        if not self._fixed_block:
            block_joint = self.sim.get_jnt_qposadr('block1joint')
            qpos_limits[block_joint:block_joint + 7] = [
                self.block_xrange,  # x
                self.block_yrange,  # y
                (.4, .921),  # z
                (0, 1),  # quat 0
                (0, 0),  # quat 1
                (0, 0),  # quat 2
                (-1, 1),  # quat 3
            ]
        return spaces.Box(*map(np.array, zip(*qpos_limits + qvel_limits)))

    def _qvel_obs(self):
        def get_qvels(joints):
            base_qvel = []
            for joint in joints:
                try:
                    base_qvel.append(self.sim.get_joint_qvel(joint))
                except RuntimeError:
                    pass
            return np.array(base_qvel)

        return get_qvels(['slide_x', 'slide_x'])

    def _get_obs(self):
        return np.concatenate([self.sim.qpos, self._qvel_obs()])

    def block_pos(self):
        return self.sim.get_body_xpos(self._block_name)

    def gripper_pos(self):
        finger1, finger2 = [self.sim.get_body_xpos(name) for name in self._finger_names]
        return (finger1 + finger2) / 2.

    def _is_successful(self):
        return self.block_pos()[2] > self.initial_block_pos[2] + self.min_lift_height

    def compute_terminal(self):
        EPSILON = .01
        below_table = self.block_pos()[2] < self.initial_qpos[2] - EPSILON
        return below_table or self._is_successful()

    def compute_reward(self):
        if self._is_successful():
            return 1
        else:
            return 0

    def step(self, action):
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


