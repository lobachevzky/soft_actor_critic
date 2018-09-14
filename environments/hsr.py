from abc import abstractmethod
from collections.__init__ import namedtuple
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from gym import spaces
from gym.utils import closer
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import mujoco
from mujoco import MujocoError, ObjType


class HSREnv:
    def __init__(self,
                 xml_filepath: Path,
                 steps_per_action: int,
                 geofence: float,
                 goal_space: spaces.Box,
                 block_space: spaces.Box,
                 min_lift_height: float = None,
                 randomize_pose: bool=False,
                 obs_type: str=None,
                 image_dimensions: Tuple[int] = None,
                 record_path: Path = None,
                 record_freq: int = None,
                 record: bool = False,
                 concat_recordings: bool = False,
                 render_freq: int = None):
        if not xml_filepath.is_absolute():
            xml_filepath = Path(Path(__file__).parent, xml_filepath)

        self.geofence = geofence
        self._obs_type = obs_type
        self.block_xrange, self.block_yrange = zip(block_space.low, block_space.high)
        self._block_name = 'block1'
        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [left_finger_name, left_finger_name.replace('_l_', '_r_')]
        self._episode = 0
        self._time_steps = 0

        # required for OpenAI code
        self.metadata = {'render.modes': 'rgb_array'}
        self.reward_range = -np.inf, np.inf
        self.spec = None
        self.render_freq = render_freq
        self.steps_per_action = steps_per_action

        # record stuff
        self._video_recorder = None
        self._concat_recordings = concat_recordings
        self._record = any((concat_recordings,
                            record_path,
                            record_freq,
                            record))
        if self._record:
            self._record_path = record_path or Path('/tmp/training-video')
            image_dimensions = image_dimensions or (1000, 1000)
            self._record_freq = record_freq or 20

            if concat_recordings:
                self._video_recorder = self.reset_recorder(self._record_path)
        else:
            image_dimensions = image_dimensions or []
        self._image_dimensions = image_dimensions

        self.sim = mujoco.Sim(str(xml_filepath), *image_dimensions, n_substeps=1)

        # initial values
        self.initial_qpos = self.sim.qpos.ravel().copy()
        self.initial_qvel = self.sim.qvel.ravel().copy()
        self.initial_block_pos = np.copy(self.block_pos())

        def adjust_dim(space: spaces.Box, offset: Tuple, dim: int):
            low_offset = np.zeros(space.shape)
            high_offset = np.zeros(space.shape)
            low_offset[dim] = offset[0]
            high_offset[dim] = offset[1]
            return spaces.Box(
                low=space.low + low_offset,
                high=space.high + high_offset,
            )

        # goal space
        if min_lift_height:
            min_lift_height += geofence
            self.goal_space = adjust_dim(space=goal_space,
                                         offset=(min_lift_height, min_lift_height),
                                         dim=2)
        else:
            self.goal_space = goal_space
        self.goal = None

        # block space
        z = self.initial_block_pos[2]
        self._block_space = spaces.Box(
            low=np.concatenate([block_space.low, [z, -1]]),
            high=np.concatenate([block_space.high, [z, 1]])
        )

        def using_joint(name):
            return self.sim.contains(ObjType.JOINT, name)

        self._base_joints = list(filter(using_joint, ['slide_x', 'slide_y']))
        raw_obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.sim.nq + len(self._base_joints),))
        self.observation_space = spaces.Tuple(Observation(
            observation=raw_obs_space,
            goal=self.goal_space
        ))

        block_joint = self.sim.get_jnt_qposadr('block1joint')
        self._block_qposadrs = block_joint + np.append(np.arange(3), 6)

        # joint space
        all_joints = ['slide_x', 'slide_y', 'arm_lift_joint', 'arm_flex_joint',
                      'wrist_roll_joint', 'hand_l_proximal_joint']
        self._joints = list(filter(using_joint, all_joints))
        jnt_range_idx = [self.sim.name2id(ObjType.JOINT, j) for j in self._joints]
        self._joint_space = spaces.Box(*map(np.array, zip(*self.sim.jnt_range[jnt_range_idx])))
        self._joint_qposadrs = [self.sim.get_jnt_qposadr(j) for j in self._joints]
        self.randomize_pose = randomize_pose
        self.init_qpos = self.sim.qpos.ravel().copy()
        self.init_qvel = self.sim.qvel.ravel().copy()
        self._image_dimensions = image_dimensions
        self._base_joints = ['slide_x', 'slide_y']
        n_base_joints = sum(
            int(self.sim.contains(ObjType.JOINT, j)) for j in self._base_joints)
        self.mujoco_obs_space = self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.sim.nq + n_base_joints, ))

        # action space
        self.action_space = spaces.Box(
            low=self.sim.actuator_ctrlrange[:-1, 0],
            high=self.sim.actuator_ctrlrange[:-1, 1],
            dtype=np.float32)

        self.initial_qpos = np.copy(self.init_qpos)
        self.initial_block_pos = np.copy(self.block_pos())

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self, mode=None, camera_name=None, labels=None):
        if mode == 'rgb_array':
            return self.sim.render_offscreen(camera_name=camera_name, labels=labels)
        return self.sim.render(camera_name=camera_name, labels=labels)

    def image(self, camera_name='rgb'):
        return self.sim.render_offscreen(camera_name)

    def _get_obs(self):
        obs = np.concatenate([self.sim.qpos, self._qvel_obs()])
        assert self.mujoco_obs_space.contains(obs)
        return obs

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

    def compute_terminal(self):
        return self._is_successful()

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

        self._time_steps += 1
        assert np.shape(action) == np.shape(self.sim.ctrl)
        self._set_action(action)
        done = self.compute_terminal()
        reward = self.compute_reward()
        if reward > 0:
            for _ in range(50):
                if self.render_freq is not None:
                    self.render()
                if self._record:
                    self._video_recorder.capture_frame()
        return self._get_obs(), reward, done, {}

    def _set_action(self, action):
        assert np.shape(action) == np.shape(self.sim.ctrl)
        for i in range(self.steps_per_action):
            self.sim.ctrl[:] = action
            self.sim.step()
            if self.render_freq is not None and i % self.render_freq == 0:
                self.render()
            if self._record and i % self._record_freq == 0:
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

        qpos[self.sim.get_jnt_qposadr('hand_r_proximal_joint')] = qpos[
            self.sim.get_jnt_qposadr('hand_l_proximal_joint')]

        # sample block_space
        block_joint = self.sim.get_jnt_qposadr('block1joint')
        qpos[block_joint + 0] = np.random.uniform(*self.block_xrange)
        qpos[block_joint + 1] = np.random.uniform(*self.block_yrange)
        qpos[block_joint + 3] = np.random.uniform(0, 1)
        qpos[block_joint + 6] = np.random.uniform(-1, 1)

        assert qpos.shape == (self.sim.nq, )
        self.sim.qpos[:] = qpos.copy()
        self.sim.qvel[:] = 0
        self.sim.forward()
        if self._time_steps > 0:
            self._episode += 1
        if self._record and not self._concat_recordings:
            if self._video_recorder:
                self._video_recorder.close()
            record_path = Path(self._record_path, str(self._episode))
            self._video_recorder = self.reset_recorder(record_path)

        self.sim.mocap_pos[:] = self.goal3d
        return self._get_obs()

    def block_pos(self):
        return self.sim.get_body_xpos(self._block_name)

    def gripper_pos(self):
        finger1, finger2 = [self.sim.get_body_xpos(name) for name in self._finger_names]
        return (finger1 + finger2) / 2.

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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.sim.__exit__()

    @property
    def goal3d(self):
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


Observation = namedtuple('Obs', 'observation goal')


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