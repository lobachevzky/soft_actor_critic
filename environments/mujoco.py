from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple
from gym.utils import closer

import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import mujoco
from mujoco import ObjType, MujocoError


class MujocoEnv:
    def __init__(self,
                 xml_filepath: Path,
                 steps_per_action: int,
                 randomize_pose=False,
                 image_dimensions: Optional[Tuple[int]] = None,
                 record_path: Optional[Path] = None,
                 record_freq: int = 0,
                 record: bool = False,
                 concat_recordings: bool = False,
                 render_freq: int = 0):
        if not xml_filepath.is_absolute():
            xml_filepath = Path(Path(__file__).parent, xml_filepath)

        self._block_name = 'block1'
        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [left_finger_name, left_finger_name.replace('_l_', '_r_')]
        self._episode = 0
        self._time_steps = 0
        self.observation_space = self.action_space = None

        # required for OpenAI code
        self.metadata = {'render.modes': 'rgb_array'}
        self.reward_range = -np.inf, np.inf
        self.spec = None
        self.render_freq = render_freq
        self.steps_per_action = steps_per_action

        self._video_recorder = None
        self._concat_recordings = concat_recordings
        self._record_video = any((concat_recordings,
                                  record_path,
                                  record_freq,
                                  record))
        if self._record_video:
            self._record_path = record_path or Path('/tmp/training-video')
            image_dimensions = image_dimensions or (1000, 1000)
            self._record_freq = record_freq or 20

            if concat_recordings:
                self._video_recorder = self.reset_recorder(self._record_path)
        else:
            image_dimensions = image_dimensions or []

        self.sim = mujoco.Sim(str(xml_filepath), *image_dimensions, n_substeps=1)

        self.randomize_pose = randomize_pose
        self.init_qpos = self.sim.qpos.ravel().copy()
        self.init_qvel = self.sim.qvel.ravel().copy()
        self._image_dimensions = image_dimensions

    def seed(self, seed=None):
        np.random.seed(seed)

    def server_values(self):
        return self.sim.qpos, self.sim.qvel

    def render(self, mode=None, camera_name=None, labels=None):
        if mode == 'rgb_array':
            return self.sim.render_offscreen(camera_name=camera_name, labels=labels)
        return self.sim.render(camera_name=camera_name, labels=labels)

    def image(self, camera_name='rgb'):
        return self.sim.render_offscreen(camera_name)

    def block_pos(self):
        return self.sim.get_body_xpos(self._block_name)

    def gripper_pos(self):
        finger1, finger2 = [self.sim.get_body_xpos(name) for name in self._finger_names]
        return (finger1 + finger2) / 2.

    def compute_terminal(self):
        return self._is_successful()

    def compute_reward(self):
        if self._is_successful():
            return 1
        else:
            return 0

    def step(self, action):
        self._time_steps += 1
        assert np.shape(action) == np.shape(self.sim.ctrl)
        self._set_action(action)
        done = self.compute_terminal()
        reward = self.compute_reward()
        if reward > 0:
            for _ in range(50):
                if self.render_freq > 0:
                    self.render()
                if self._record_video:
                    self._video_recorder.capture_frame()
        return self._get_obs(), reward, done, {}

    def _set_action(self, action):
        assert np.shape(action) == np.shape(self.sim.ctrl)
        for i in range(self.steps_per_action):
            self.sim.ctrl[:] = action
            self.sim.step()
            if self.render_freq > 0 and i % self.render_freq == 0:
                self.render()
            if self._record_video and i % self._record_freq == 0:
                self._video_recorder.capture_frame()

    def reset(self):
        self.sim.reset()
        qpos = np.copy(self.init_qpos)
        if self.randomize_pose:
            for joint in [
                'slide_x', 'slide_y', 'arm_lift_joint', 'arm_flex_joint',
                'wrist_roll_joint', 'hand_l_proximal_joint'
            ]:
                try:
                    qpos_idx = self.sim.get_jnt_qposadr(joint)
                except MujocoError:
                    continue
                jnt_range_idx = self.sim.name2id(ObjType.JOINT, joint)
                qpos[qpos_idx] = \
                    np.random.uniform(
                        *self.sim.jnt_range[jnt_range_idx])
                # self.sim.jnt_range[jnt_range_idx][1]

        qpos[
            self.sim.get_jnt_qposadr('hand_r_proximal_joint')] = qpos[
            self.sim.get_jnt_qposadr('hand_l_proximal_joint')]
        qpos = self._reset_qpos(qpos)
        assert qpos.shape == (self.sim.nq, )
        self.sim.qpos[:] = qpos.copy()
        self.sim.qvel[:] = 0
        self.sim.forward()
        if self._time_steps > 0:
            self._episode += 1
        if self._record_video and not self._concat_recordings:
            if self._video_recorder:
                self._video_recorder.close()
            record_path = Path(self._record_path, str(self._episode))
            self._video_recorder = self.reset_recorder(record_path)
        return self._get_obs()

    def reset_recorder(self, record_path: Path):
        record_path.mkdir(parents=True, exist_ok=True)
        print(f'Recording video to {record_path}.mp4')
        video_recorder = VideoRecorder(
            env=self,
            base_path=str(record_path),
            metadata={'episode': self._episode},
            enabled=True,
        )
        closer.Closer().register(video_recorder)
        return video_recorder

    @abstractmethod
    def _reset_qpos(self):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.sim.__exit__()

    @abstractmethod
    def _get_obs(self):
        raise NotImplementedError

    @abstractmethod
    def _is_successful(self):
        raise NotImplementedError


def quaternion2euler(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    euler_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = 1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    euler_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    euler_z = np.arctan2(t3, t4)

    return euler_x, euler_y, euler_z


def distance_between(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1))


def escaped(pos, world_upper_bound, world_lower_bound):
    # noinspection PyTypeChecker
    return np.any(pos > world_upper_bound) \
           or np.any(pos < world_lower_bound)


def get_limits(pos, size):
    return pos + size, pos - size


def point_inside_object(point, object):
    pos, size = object
    tl = pos - size
    br = pos + size
    return (tl[0] <= point[0] <= br[0]) and (tl[1] <= point[1] <= br[1])


def print1(*strings):
    print('\r', *strings, end='')
