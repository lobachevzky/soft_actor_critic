import itertools
import pickle
import time
from pathlib import Path
from typing import Optional, Tuple

import gym
import numpy as np
import tensorflow as tf
from collections import Counter
from gym import spaces

from environments.hindsight_wrapper import HindsightWrapper
from environments.unsupervised import UnsupervisedEnv
from sac.agent import AbstractAgent
from sac.policies import CategoricalPolicy, GaussianPolicy
from sac.replay_buffer import ReplayBuffer
from sac.utils import State, Step


class Trainer:
    def __init__(self, env: gym.Env, seed: Optional[int], buffer_size: int,
                 batch_size: int, num_train_steps: int, mimic_dir: Optional[str],
                 logdir: str, save_path: str, load_path: str, render: bool, **kwargs):

        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
            env.seed(seed)

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.env = env
        self.buffer = ReplayBuffer(buffer_size)
        self.render = render

        if mimic_dir:
            for path in Path(mimic_dir).iterdir():
                if path.suffix == '.pkl':
                    with Path(path).open('rb') as f:
                        self.buffer.extend(pickle.load(f))
                print('Loaded mimic file {} into buffer.'.format(path))

        self.agent = agent = self.build_agent(**kwargs)
        saver = tf.train.Saver()
        tb_writer = None
        if load_path:
            saver.restore(agent.sess, load_path)
            print("Model restored from", load_path)
        if logdir:
            tb_writer = tf.summary.FileWriter(logdir=logdir, graph=agent.sess.graph)

        count = Counter(reward=0, episode=0)
        self.episode_count = self.episode_mean = Counter()
        tick = time.time()
        s1 = self.reset()

        for timesteps in itertools.count():
            is_eval_period = timesteps % 100 == 0 or load_path is not None
            s2, r, t, i = self.step(s1=s1, is_eval_period=is_eval_period)
            s1 = s2

            count += Counter(reward=r)
            self.episode_mean.update(Counter(fps=1 / float(time.time() - tick)))
            tick = time.time()
            if save_path and timesteps % 5000 == 0:
                print("model saved in path:", saver.save(agent.sess, save_path=save_path))
            if t:
                s1 = self.reset()
                count += Counter(episode=1)
                print('({}) Episode {}\t Time Steps: {}\t Reward: {}'.format(
                    'EVAL' if is_eval_period else 'TRAIN', count['episode'],
                    timesteps, self.episode_count['reward']))
                if logdir:
                    summary = tf.Summary()
                    if is_eval_period:
                        summary.value.add(tag='eval reward',
                                          simple_value=self.episode_count['reward'])
                    for k in self.episode_count:
                        summary.value.add(tag=k, simple_value=self.episode_count[k])
                    episode_timesteps = self.episode_count['timesteps']
                    for k in self.episode_mean:
                        summary.value.add(
                            tag=k,
                            simple_value=self.episode_mean[k] / float(episode_timesteps))
                    tb_writer.add_summary(summary, timesteps)
                    tb_writer.flush()

                # zero out counters
                self.episode_count = self.episode_mean = Counter()

    def step(self, is_eval_period, s1):
        a = self.agent.get_actions(self.vectorize_state(s1), sample=(not is_eval_period))
        if self.render:
            self.env.render()
        s2, r, t, info = self.step_env(a)
        if 'log count' in info:
            self.episode_count.update(Counter(info['log count']))
        if 'log mean' in info:
            self.episode_mean.update(Counter(info['log mean']))
        if not is_eval_period:
            self.add_to_buffer(Step(s1=s1, a=a, r=r, s2=s2, t=t))
            if self.buffer_full():
                for i in range(self.num_train_steps):
                    step = self.agent.train_step(self.sample_buffer())
                    self.episode_mean.update(
                        Counter({
                                    k: getattr(step, k.replace(' ', '_'))
                                    for k in [
                                        'entropy',
                                        'V loss',
                                        'Q loss',
                                        'pi loss',
                                        'V grad',
                                        'Q grad',
                                        'pi grad',
                                    ]
                                    }))
        self.episode_count.update(Counter(reward=r, timesteps=1))
        return s2, r, t, info

    def build_agent(self, base_agent: AbstractAgent = AbstractAgent, **kwargs):
        state_shape = self.env.observation_space.shape
        if isinstance(self.env.action_space, spaces.Discrete):
            action_shape = [self.env.action_space.n]
            policy_type = CategoricalPolicy
        else:
            action_shape = self.env.action_space.shape
            policy_type = GaussianPolicy

        class Agent(policy_type, base_agent):
            def __init__(self, s_shape, a_shape):
                super(Agent, self).__init__(
                    s_shape=s_shape,
                    a_shape=a_shape,
                    **kwargs)

        return Agent(state_shape, action_shape)

    def reset(self) -> State:
        return self.env.reset()

    def step_env(self, action: np.ndarray) -> Tuple[State, float, bool, dict]:
        """ Preprocess action before feeding to env """
        if type(self.env.action_space) is spaces.Discrete:
            # noinspection PyTypeChecker
            return self.env.step(np.argmax(action))
        else:
            action = np.tanh(action)
            hi, lo = self.env.action_space.high, self.env.action_space.low
            # noinspection PyTypeChecker
            return self.env.step((action + 1) / 2 * (hi - lo) + lo)

    def vectorize_state(self, state: State) -> np.ndarray:
        """ Preprocess state before feeding to network """
        return state

    def add_to_buffer(self, step: Step) -> None:
        assert isinstance(step, Step)
        self.buffer.append(step)

    def buffer_full(self):
        return len(self.buffer) >= self.batch_size

    def sample_buffer(self):
        step = Step(*self.buffer.sample(self.batch_size))
        return step.replace(
            s1=self.vectorize_state(step.s1),
            s2=self.vectorize_state(step.s2),
        )


class TrajectoryTrainer(Trainer):
    def __init__(self, mimic_save_dir: Optional[str], **kwargs):
        self.mimic_save_dir = mimic_save_dir
        if mimic_save_dir is not None:
            path = Path(mimic_save_dir)
            print('Using model dir', path.absolute())
            path.mkdir(parents=True, exist_ok=True)
        self.mimic_num = 0
        self.trajectory_start = None
        super().__init__(**kwargs)
        self.s1 = self.reset()

    def reset(self) -> State:
        if self.mimic_save_dir is not None:
            path = Path(self.mimic_save_dir, str(self.mimic_num) + '.pkl')
            trajectory = list(self.buffer[self.trajectory_start: self.buffer.pos])
            with path.open(mode='wb') as f:
                pickle.dump(trajectory, f)
            self.mimic_num += 1
        self.trajectory_start = self.buffer.pos
        self.s1 = super().reset()
        return self.s1

    def get_nth_step(self, n: int):
        return self.buffer[(self.trajectory_start + n) % self.buffer.maxlen]


class HindsightTrainer(TrajectoryTrainer):
    def __init__(self, env: HindsightWrapper, n_goals: int, **kwargs):
        self.n_goals = n_goals
        assert isinstance(env, HindsightWrapper)
        super().__init__(env=env, **kwargs)

    def add_hindsight_trajectories(self) -> None:
        assert isinstance(self.env, HindsightWrapper)
        start = self.trajectory_start
        stop = self.buffer.pos
        if start and stop and stop > start:
            trajectory = self.buffer[start: stop]
            final_state = self.get_nth_step(stop - start - 1)
            for step in self.env.recompute_trajectory(trajectory, final_state):
                self.add_to_buffer(step)
            if self.n_goals - 1 > 0:
                stop_indexes = np.random.randint(1, stop - start - 1, size=self.n_goals - 1)
                assert isinstance(stop_indexes, np.ndarray)
                for stop in stop_indexes:
                    trajectory = self.buffer[start: stop]
                    final_state = self.get_nth_step(stop)
                    for step in self.env.recompute_trajectory(trajectory,
                                                              final_step=final_state):
                        self.add_to_buffer(step)

    def reset(self) -> State:
        self.add_hindsight_trajectories()
        return super().reset()

    def vectorize_state(self, state: State) -> np.ndarray:
        assert isinstance(self.env, HindsightWrapper)
        return self.env.vectorize_state(state)
