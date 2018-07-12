import itertools
import time
from collections import Counter, namedtuple
from typing import Optional, Tuple

import gym
import numpy as np
import tensorflow as tf
from gym import Wrapper, spaces

from environments.hindsight_wrapper import HindsightWrapper
from environments.multi_task import MultiTaskEnv
from sac.agent import AbstractAgent
from sac.policies import CategoricalPolicy, GaussianPolicy
from sac.replay_buffer import ReplayBuffer
from sac.utils import Obs, Step, normalize, unwrap_env, vectorize

Agents = namedtuple('Agents', 'train act')


class Trainer:
    def __init__(self, env: gym.Env, seed: Optional[int],
                 buffer_size: int, batch_size: int, seq_len: int, num_train_steps: int,
                 logdir: str, save_path: str, load_path: str, render: bool, **kwargs):

        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
            env.seed(seed)

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.env = env
        self.buffer = ReplayBuffer(buffer_size)
        self.save_path = save_path

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.inter_op_parallelism_threads = 1
        sess = tf.Session(config=config)

        self.agent = agent = self.build_agent(
            sess=sess, batch_size=None, reuse=False, **kwargs)
        self.seq_len = self.agent.seq_len
        saver = tf.train.Saver()
        tb_writer = None
        if load_path:
            saver.restore(sess, load_path)
            print("Model restored from", load_path)
        if logdir:
            tb_writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)

        self.count = Counter(reward=0, episode=0)
        self.episode_count = Counter()

        obs = self.reset()
        if not isinstance(obs, np.ndarray):
            self.preprocess_func = None

            if not isinstance(obs, np.ndarray):
                env = self.env

            while self.preprocess_func is None:
                try:
                    self.preprocess_func = env.preprocess_obs
                except AttributeError:
                    try:
                        env = env.env
                    except AttributeError:
                        self.preprocess_func = vectorize

        for episodes in itertools.count(1):
            if save_path and episodes % 25 == 1:
                print("model saved in path:", saver.save(sess, save_path=save_path))
                saver.save(sess, save_path.replace('<episode>', str(episodes)))
            self.episode_count = self.run_episode(
                o1=self.reset(),
                render=render,
                perform_updates=not self.is_eval_period() and load_path is None)

            episode_reward = self.episode_count['reward']
            self.count.update(
                Counter(reward=episode_reward, episode=1, time_steps=self.time_steps()))
            print('({}) Episode {}\t Time Steps: {}\t Reward: {}'.format(
                'EVAL' if self.is_eval_period() else 'TRAIN', episodes,
                self.count['time_steps'], episode_reward))
            if logdir:
                summary = tf.Summary()
                if self.is_eval_period():
                    summary.value.add(tag='eval reward', simple_value=episode_reward)
                else:
                    for k in self.episode_count:
                        summary.value.add(tag=k, simple_value=self.episode_count[k])
                tb_writer.add_summary(summary, self.count['time_steps'])
                tb_writer.flush()

    def is_eval_period(self):
        return self.count['episode'] % 100 == 99

    def trajectory(self, final_index=None) -> Optional[Step]:
        if final_index is None:
            final_index = 0  # points to current time step
        else:
            final_index -= self.time_steps()
        if self.buffer.empty:
            return None
        return Step(*self.buffer[-self.time_steps():final_index])

    def time_steps(self):
        return self.episode_count['time_steps']

    def run_episode(self, o1, perform_updates, render):
        self.episode_count = Counter()
        episode_mean = Counter()
        tick = time.time()
        s = self.agent.initial_state
        for time_steps in itertools.count(1):
            a, s = self.agent.get_actions(
                self.preprocess_obs(o1), state=s, sample=(not self.is_eval_period()))
            if render:
                self.env.render()
            o2, r, t, info = self.step(a)
            if 'print' in info:
                print('Time step:', time_steps, info['print'])
            if 'log count' in info:
                self.episode_count.update(Counter(info['log count']))
            if 'log mean' in info:
                episode_mean.update(Counter(info['log mean']))
            self.add_to_buffer(Step(s=s, o1=o1, a=a, r=r, o2=o2, t=t))
            self.episode_count.update(Counter(reward=r, time_steps=1))

            if self.buffer_full() and perform_updates:
                for i in range(self.num_train_steps):
                    step = self.agent.train_step(self.sample_buffer())
                    episode_mean.update(
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
            o1 = o2
            episode_mean.update(Counter(fps=1 / float(time.time() - tick)))
            tick = time.time()
            if t:
                for k in episode_mean:
                    self.episode_count[k] = episode_mean[k] / float(time_steps)
                return self.episode_count

    def build_agent(self, base_agent: AbstractAgent, **kwargs):
        state_shape = self.env.observation_space.shape
        if isinstance(self.env.action_space, spaces.Discrete):
            action_shape = [self.env.action_space.n]
            policy_type = CategoricalPolicy
        else:
            action_shape = self.env.action_space.shape
            policy_type = GaussianPolicy

        class Agent(policy_type, base_agent):
            def __init__(self):
                super(Agent, self).__init__(
                    o_shape=state_shape,
                    a_shape=action_shape,
                    **kwargs)

        return Agent()

    def reset(self) -> Obs:
        return self.env.reset()

    def step(self, action: np.ndarray) -> Tuple[Obs, float, bool, dict]:
        """ Preprocess action before feeding to env """
        if type(self.env.action_space) is spaces.Discrete:
            # noinspection PyTypeChecker
            return self.env.step(np.argmax(action))
        else:
            action = np.tanh(action)
            hi, lo = self.env.action_space.high, self.env.action_space.low
            # noinspection PyTypeChecker
            return self.env.step((action + 1) / 2 * (hi - lo) + lo)

    def preprocess_obs(self, obs, shape: Optional[tuple] = None):
        if self.preprocess_func is not None:
            obs = self.preprocess_func(obs, shape)
        return normalize(
            vector=obs,
            low=self.env.observation_space.low,
            high=self.env.observation_space.high)

    def add_to_buffer(self, step: Step) -> None:
        assert isinstance(step, Step)
        self.buffer.append(step)

    def buffer_full(self):
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

    def time_steps(self):
        return self.episode_count['time_steps']


class HindsightTrainer(Trainer):
    def __init__(self, env: HindsightWrapper, n_goals: int, **kwargs):
        self.n_goals = n_goals
        assert isinstance(env, HindsightWrapper)
        super().__init__(env=env, **kwargs)

    def add_hindsight_trajectories(self) -> None:
        if self.trajectory():
            new_trajectory = self.trajectory()
            if self.time_steps() > 0:
                new_recomputed_trajectory = self.env.recompute_trajectory(new_trajectory)
                assert bool(Step(*new_recomputed_trajectory[-1]).t) is True
                self.buffer.append(self.env.recompute_trajectory(new_trajectory))
                assert bool(Step(*self.buffer[-1]).t) is True
                if self.n_goals - 1 and self.time_steps() > 0:
                    final_indexes = np.random.randint(
                        1, self.time_steps(), size=self.n_goals - 1)
                    assert isinstance(final_indexes, np.ndarray)

                    for final_index in final_indexes:
                        self.buffer.append(
                            self.env.recompute_trajectory(
                                self.trajectory()[:final_index]))

    def reset(self) -> Obs:
        self.add_hindsight_trajectories()
        return super().reset()

