import mujoco_py
import os
from abc import abstractmethod
from copy import deepcopy, copy

import numpy as np

from environments.base import BaseEnv


class MujocoEnv(BaseEnv):
    def __init__(self,
                 xml_filepath,
                 image_dimensions,
                 neg_reward,
                 steps_per_action,
                 frames_per_step=20):
        fullpath = os.path.join(os.path.dirname(__file__), xml_filepath)
        if not fullpath.startswith("/"):
            fullpath = os.path.join(os.path.dirname(__file__), "assets", fullpath)
        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=frames_per_step)
        self.sim.forward()
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.initial_state = deepcopy(self.sim.get_state())
        super().__init__(
            image_dimensions=image_dimensions,
            neg_reward=neg_reward,
            steps_per_action=steps_per_action)

    def server_values(self):
        return self.sim.qpos, self.sim.qvel

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        return self.viewer

    def render(self, mode=None, camera_name=None, labels=None):
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def image(self, camera_name='rgb'):
        return self.sim.render_offscreen(*self._image_dimensions, camera_name)

    def step(self, action):
        assert np.shape(action) == np.shape(self.sim.data.ctrl)
        return super().step(action)

    def _set_action(self, action):
        assert np.shape(action) == np.shape(self.sim.data.ctrl)
        self.sim.data.ctrl[:] = action
        self.sim.step()

    def reset(self):
        self.sim.reset()
        self._step_num = 0

        self._set_new_goal()
        qpos = self.reset_qpos()
        qvel = self.initial_state.qvel + \
            np.random.uniform(size=self.sim.model.nv, low=-0.01, high=0.01)
        assert qpos.shape == (self.sim.model.nq, ) and qvel.shape == (self.sim.model.nv, )
        self.sim.data.qpos[:] = qpos.copy()
        self.sim.data.qvel[:] = qvel.copy()
        self.sim.forward()
        return self._get_obs()

    @abstractmethod
    def reset_qpos(self):
        raise NotImplemented

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.sim.__exit__()
