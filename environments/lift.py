import numpy as np
from gym import spaces

from environments.mujoco import MujocoEnv


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
                 min_lift_height=.08,
                 geofence=.05,
                 cheat_prob=0,
                 **kwargs):
        if block_xrange is None:
            block_xrange = (0, 0)
        if block_yrange is None:
            block_yrange = (0, 0)
        self.block_xrange = block_xrange
        self.block_yrange = block_yrange
        self.min_lift_height = min_lift_height + geofence
        self._goal = None

        super().__init__(**kwargs)

        self._table_height = self.sim.get_body_xpos('pan')[2]
        self._rotation_actuators = ["arm_flex_motor"]  # , "wrist_roll_motor"]
        self.unwrapped = self

    @property
    def goal(self):
        return self._goal

    @property
    def goal3d(self):
        return self._goal

    def reset(self):
        obs = super().reset()
        self._goal = self.block_pos() + np.array([0, 0, self.min_lift_height])
        self.sim.mocap_pos[:] = self.goal3d
        return obs

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

    def _is_successful(self):
        return self.block_pos()[2] > self.initial_block_pos[2] + self.min_lift_height
