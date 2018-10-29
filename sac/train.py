import itertools
import time
from collections import Counter, deque, namedtuple
from pathlib import Path
from typing import Optional, Tuple

import gym
import numpy as np
import tensorflow as tf
from gym import Wrapper, spaces

from environments.hindsight_wrapper import HindsightWrapper
from sac.agent import AbstractAgent
from sac.policies import CategoricalPolicy, GaussianPolicy
from sac.replay_buffer import ReplayBuffer
from sac.utils import (Obs, Step, create_sess, get_space_attrs, unwrap_env, vectorize)

Agents = namedtuple('Agents', 'train act')


class Trainer:
    def __init__(self,
                 env: gym.Env,
                 seed: Optional[int],
                 buffer_size: int,
                 batch_size: int,
                 seq_len: int,
                 n_train_steps: int,
                 sess: tf.Session = None,
                 preprocess_func=None,
                 action_space=None,
                 observation_space=None,
                 **kwargs):

        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
            env.seed(seed)

        self.episodes = None
        self.episode_count = None
        self.n_train_steps = n_train_steps
        self.batch_size = batch_size
        self.env = env
        self.buffer = ReplayBuffer(buffer_size)
        self.sess = sess or create_sess()
        self.action_space = action_space or env.action_space
        observation_space = observation_space or env.observation_space

        obs = env.reset()
        self.preprocess_func = preprocess_func
        if preprocess_func is None and not isinstance(obs, np.ndarray):
            try:
                self.preprocess_func = unwrap_env(
                    env, lambda e: hasattr(e, 'preprocess_obs')).preprocess_obs
            except RuntimeError:
                self.preprocess_func = vectorize


        observation_space = spaces.Box(
            *[
                self.preprocess_obs(get_space_attrs(observation_space, attr))
                for attr in ['low', 'high']
            ],
            dtype=np.float32)

        self.agents = Agents(
            act=self.build_agent(
                sess=self.sess,
                batch_size=None,
                seq_len=1,
                reuse=False,
                action_space=action_space,
                observation_space=observation_space,
                **kwargs),
            train=self.build_agent(
                sess=self.sess,
                batch_size=batch_size,
                seq_len=seq_len,
                reuse=True,
                action_space=action_space,
                observation_space=observation_space,
                **kwargs))
        self.seq_len = self.agents.act.seq_len

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.episode_time_step = tf.placeholder(tf.int32, name='episode_time_steps')
        self.increment_global_step = tf.assign_add(self.global_step,
                                                   self.episode_time_step)
        self.sess.run(self.global_step.initializer)

        # self.train(load_path, logdir, render, save_path)

    def train(self,
              save_path: Path,
              load_path: Path,
              logdir: Path,
              render: bool,
              save_threshold: int = None):
        saver = tf.train.Saver()
        tb_writer = None
        if load_path:
            saver.restore(self.sess, load_path)
            print("Model restored from", load_path)
        if logdir:
            tb_writer = tf.summary.FileWriter(logdir=logdir, graph=self.sess.graph)

        past_rewards = deque(maxlen=save_threshold)
        best_average = -np.inf

        for episodes in itertools.count(1):
            self.episodes = episodes
            self.episode_count = self.run_episode(
                o1=self.reset(),
                render=render,
                eval_period=self.is_eval_period() and load_path is None)

            episode_reward = self.episode_count['reward']

            # save model
            passes_save_threshold = True
            if save_threshold is not None:
                past_rewards.append(episode_reward)
                new_average = np.mean(past_rewards)
                if new_average < best_average:
                    passes_save_threshold = False
                else:
                    best_average = new_average

            if save_path and episodes % 10 == 1 and passes_save_threshold:
                print("model saved in path:", saver.save(self.sess, save_path=save_path))
                saver.save(self.sess, save_path.replace('<episode>', str(episodes)))

            time_steps, _ = self.sess.run(
                [self.global_step, self.increment_global_step],
                {self.episode_time_step: self.episode_count['time_steps']})
            print('({}) Episode {}\t Time Steps: {}\t Reward: {}'.format(
                'EVAL' if self.is_eval_period() else 'TRAIN', episodes, time_steps,
                episode_reward))

            if logdir:
                summary = tf.Summary()
                if self.is_eval_period():
                    summary.value.add(tag='eval reward', simple_value=episode_reward)
                else:
                    for k in self.episode_count:
                        summary.value.add(
                            tag=k.replace('_', ' '), simple_value=self.episode_count[k])
                tb_writer.add_summary(summary, time_steps)
                tb_writer.flush()

    def is_eval_period(self):
        return self.episodes % 100 == 0

    def trajectory(self, time_steps: int, final_index=None) -> Optional[Step]:
        if final_index is None:
            final_index = 0  # points to current time step
        else:
            final_index -= time_steps  # relative to start of episode
        if self.buffer.empty:
            return None
        return Step(*self.buffer[-time_steps:final_index])

    def run_episode(self, o1, eval_period, render):
        episode_count = Counter()
        episode_mean = Counter()
        tick = time.time()
        s = self.agents.act.initial_state
        for time_steps in itertools.count(1):
            a, s = self.get_actions(o1, s, sample=not eval_period)
            o2, r, t, info = self.step(a, render)
            if 'print' in info:
                print('Time step:', time_steps, info['print'])
            if not eval_period:
                episode_mean.update(self.perform_update())
            self.add_to_buffer(Step(s=s, o1=o1, a=a, r=r, o2=o2, t=t))
            o1 = o2
            # noinspection PyTypeChecker
            episode_mean.update(
                Counter(fps=1 / float(time.time() - tick), **info.get('log mean', {})))
            # noinspection PyTypeChecker
            episode_count.update(
                Counter(reward=r, time_steps=1, **info.get('log count', {})))
            tick = time.time()
            if t:
                for k in episode_mean:
                    episode_count[k] = episode_mean[k] / float(time_steps)
                return episode_count

    def train_step(self, add_fetch: list = None):
        return self.agents.act.train_step(self.sample_buffer(),
                                          add_fetch=add_fetch)

    def perform_update(self):
        counter = Counter()
        if self.buffer_ready():
            for i in range(self.n_train_steps):
                counter.update(
                    Counter({
                        k.replace(' ', '_'): v
                        for k, v in self.train_step().items() if np.isscalar(v)
                    }))
        return counter

    def get_actions(self, o1, s, sample: bool):
        obs = self.preprocess_obs(o1)
        # assert self.observation_space.contains(obs)
        return self.agents.act.get_actions(o=obs, state=s, sample=sample)

    def build_agent(self,
                    base_agent: AbstractAgent,
                    action_space=None,
                    observation_space=None,
                    **kwargs):
        if observation_space is None:
            observation_space = self.observation_space
        if action_space is None:
            action_space = self.action_space

        def space_to_size(space: gym.Space):
            if isinstance(space, spaces.Discrete):
                return space.n
            elif isinstance(space, (spaces.Dict, spaces.Tuple)):
                if isinstance(space, spaces.Dict):
                    _spaces = list(space.spaces.values())
                else:
                    _spaces = list(space.spaces)
                return sum(space_to_size(s) for s in _spaces)
            else:
                return space.shape[0]

        if isinstance(action_space, spaces.Discrete):
            policy_type = CategoricalPolicy
        else:
            policy_type = GaussianPolicy

        class Agent(policy_type, base_agent):
            def __init__(self):
                super(Agent, self).__init__(
                    o_shape=observation_space.shape,
                    a_shape=[space_to_size(action_space)],
                    **kwargs)

        return Agent()

    def reset(self) -> Obs:
        self.episode_count = None
        return self.env.reset()

    def step(self, action: np.ndarray, render: bool) -> Tuple[Obs, float, bool, dict]:
        """ Preprocess action before feeding to env """
        if render:
            self.env.render()
        if type(self.action_space) is spaces.Discrete:
            # noinspection PyTypeChecker
            return self.env.step(np.argmax(action))
        else:
            return self.env.step(fit_to_space(action, space=self.action_space))

    def preprocess_obs(self, obs, shape: tuple = None):
        if self.preprocess_func is not None:
            obs = self.preprocess_func(obs, shape)
        return obs

    def add_to_buffer(self, step: Step) -> None:
        assert isinstance(step, Step)
        self.buffer.append(step)

    def buffer_ready(self):
        return len(self.buffer) >= self.batch_size

    def sample_buffer(self) -> Step:
        sample = Step(*self.buffer.sample(self.batch_size, seq_len=self.seq_len))
        if self.seq_len is None:
            # leave state as dummy value for non-recurrent
            shape = [self.batch_size, -1]
            return Step(
                o1=self.preprocess_obs(sample.o1, shape=shape),
                o2=self.preprocess_obs(sample.o2, shape=shape),
                s=sample.s,
                a=sample.a,
                r=sample.r,
                t=sample.t)
        else:
            # adjust state for recurrent networks
            shape = [self.batch_size, self.seq_len, -1]
            return Step(
                o1=self.preprocess_obs(sample.o1, shape=shape),
                o2=self.preprocess_obs(sample.o2, shape=shape),
                s=np.swapaxes(sample.s[:, -1], 0, 1),
                a=sample.a[:, -1],
                r=sample.r[:, -1],
                t=sample.t[:, -1])


class HindsightTrainer(Trainer):
    def __init__(self, env: Wrapper, n_goals: int, **kwargs):
        self.n_goals = n_goals
        self.hindsight_env = unwrap_env(env, lambda e: isinstance(e, HindsightWrapper))
        assert isinstance(self.hindsight_env, HindsightWrapper)
        super().__init__(env=env, **kwargs)

    def add_hindsight_trajectories(self, time_steps: int) -> None:
        assert isinstance(self.hindsight_env, HindsightWrapper)
        if time_steps > 0:
            trajectory = self.trajectory(time_steps=time_steps)
            new_trajectory = self.hindsight_env.recompute_trajectory(trajectory)
            self.buffer.append(new_trajectory)
        if self.n_goals - 1 and time_steps > 1:
            final_indexes = np.random.randint(1, time_steps, size=self.n_goals - 1)
            assert isinstance(final_indexes, np.ndarray)

            for final_index in final_indexes:
                traj = self.trajectory(time_steps=time_steps, final_index=final_index)
                new_traj = self.hindsight_env.recompute_trajectory(traj)
                self.buffer.append(new_traj)

    def reset(self) -> Obs:
        time_steps = 0 if self.episode_count is None else self.episode_count['time_steps']
        self.add_hindsight_trajectories(time_steps)
        return super().reset()


def fit_to_space(x: np.ndarray, space: spaces.Box) -> np.ndarray:
    return (np.tanh(x) + 1) / 2 * (space.high - space.low) + space.low
