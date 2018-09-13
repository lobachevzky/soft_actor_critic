import itertools
from collections import namedtuple

import numpy as np
from gym import spaces

from environments.lift import LiftEnv
from environments.mujoco import distance_between, MujocoEnv

Observation = namedtuple('Obs', 'observation goal')


class ShiftEnv(MujocoEnv):
    def __init__(self,
                 geofence: float,
                 fixed_block=None,
                 fixed_goal=None,
                 goal_x=(-.11, .09),
                 goal_y=(-.19, .2),
                 obs_type=None,
                 **kwargs):
        self._obs_type = obs_type
        self.fixed_block = fixed_block
        self.fixed_goal = fixed_goal
        self.geofence = geofence
        self._goal_space = spaces.Box(*map(np.array, zip(goal_x, goal_y)))
        self._goal = self.goal_space.sample() if fixed_goal is None else fixed_goal
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

    @property
    def goal_space(self):
        return self._goal_space

    @property
    def goal(self):
        return self._goal

    @property
    def goal3d(self):
        return np.append(self._goal, self.initial_block_pos[2])

    def _is_successful(self):
        return distance_between(self.goal, self.block_pos()[:2]) < self.geofence

    def _get_obs(self):
        if self._obs_type == 'openai':

            # positions
            grip_pos = self.gripper_pos()
            dt = self.sim.nsubsteps * self.sim.timestep
            object_pos = self.block_pos()
            grip_velp = .5 * sum(self.sim.get_body_xvelp(name) for name in self._finger_names)
            # rotations
            object_rot = mat2euler(self.sim.get_body_xmat(self._block_name))

            # velocities
            object_velp = self.sim.get_body_xvelp(self._block_name) * dt
            object_velr = self.sim.get_body_xvelr(self._block_name) * dt

            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
            gripper_state = np.array(
                [self.sim.get_joint_qpos(f'hand_{x}_proximal_joint') for x in 'lr'])
            qvels = np.array(
                [self.sim.get_joint_qvel(f'hand_{x}_proximal_joint') for x in 'lr'])
            gripper_vel = dt * .5 * qvels

            obs = np.concatenate([
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ])
        else:
            obs = super()._get_obs()
        return Observation(observation=obs, goal=self.goal)

    def _reset_qpos(self, qpos):
        block_joint = self.sim.get_jnt_qposadr('block1joint')
        if self.fixed_block is None:
            qpos[[
                block_joint + 0, block_joint + 1, block_joint + 3, block_joint + 6
            ]] = np.random.uniform(
                low=list(self.goal_space.low) + [0, -1],
                high=list(self.goal_space.high) + [1, 1])
        else:
            qpos[[block_joint + 0, block_joint + 1,
                  block_joint + 2]] = self.fixed_block
        return qpos

    def reset(self):
        if self.fixed_goal is None:
            self._goal = self.goal_space.sample()
        return super().reset()

    def render(self, labels=None, **kwargs):
        if labels is None:
            labels = dict()
        # z = (.4,)
        # labels[tuple(self.goal_space.low) + z] = '['
        # labels[tuple(self.goal_space.high) + z] = ']'
        # labels[tuple(self.goal) + z] = '|'
        return super().render(labels=labels, **kwargs)


def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > np.finfo(np.float64).eps * 4.
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition, -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition, -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0)
    return euler
