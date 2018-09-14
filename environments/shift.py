import itertools

import numpy as np
from gym import spaces

from environments.mujoco import MujocoEnv, mat2euler, Observation


class ShiftEnv(MujocoEnv):
    def __init__(self,
                 geofence: float,
                 obs_type=None,
                 **kwargs):
        self._obs_type = obs_type
        self.geofence = geofence
        super().__init__(**kwargs)
        # low=np.array([-.14, -.22, .40]), high=np.array([.11, .22, .63]))
        # goal_size = np.array([.0317, .0635, .0234]) * geofence
        intervals = [2, 3, 1]
        x, y = [
            np.linspace(l, h, n)
            for l, h, n in zip(self.goal_space.low, self.goal_space.high, intervals)
        ]
        goal_corners = np.array(list(itertools.product(x, y)))
        self.labels = {tuple(g) + (.41, ): '.' for g in goal_corners}
        self.observation_space = spaces.Dict(Observation(
            observation=(spaces.Box(-np.inf, np.inf, np.shape(self._get_raw_obs()))),
            goal=self.goal_space
        )._asdict())

    def _get_raw_obs(self):
        if self._obs_type == 'openai':

            # positions
            grip_pos = self.gripper_pos()
            dt = self.sim.nsubsteps * self.sim.timestep
            object_pos = self.block_pos()
            grip_velp = .5 * sum(self.sim.get_body_xvelp(name)
                                 for name in self._finger_names)
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

            return np.concatenate([
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
            return super()._get_obs()

    def _get_obs(self):
        observation = Observation(observation=self._get_raw_obs(), goal=self.goal)._asdict()
        assert self.observation_space.contains(observation)
        return observation


