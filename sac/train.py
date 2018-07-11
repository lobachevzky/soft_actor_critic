import itertools
import time
from collections import Counter, namedtuple
from typing import Any, Iterable, Optional, Tuple

import gym
import numpy as np
import tensorflow as tf
from gym import spaces

from environments.hindsight_wrapper import HindsightWrapper
from sac.agent import AbstractAgent
from sac.policies import CategoricalPolicy, GaussianPolicy
from sac.replay_buffer import ReplayBuffer
from sac.utils import Step, Obs

Agents = namedtuple('Agents', 'train act')


class Trainer:
    def __init__(self, env: gym.Env, seed: Optional[int],
                 buffer_size: int, batch_size: int, num_train_steps: int,
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

        count = Counter(reward=0, episode=0)
        self.episode_count = episode_count = Counter()
        self.episode_mean = episode_mean = Counter()
        tick = time.time()

        s1 = self.reset()

        for time_steps in itertools.count():
            is_eval_period = count['episode'] % 100 == 99
            a = agent.get_actions(
                [self.vectorize_state(s1)], sample=(not is_eval_period)).output
            if render:
                env.render()
            s2, r, t, info = self.step(a)
            if 'print' in info:
                print('Time step:', time_steps, info['print'])
            if 'log count' in info:
                episode_count.update(Counter(info['log count']))
            if 'log mean' in info:
                episode_mean.update(Counter(info['log mean']))

            if save_path and time_steps % 5000 == 0:
                print("model saved in path:", saver.save(agent.sess, save_path=save_path))
            self.add_to_buffer(Step(s=0, o1=s1, a=a, r=r, o2=s2, t=t))
            if self.buffer_full() and not load_path:
                for i in range(self.num_train_steps):
                    sample_steps = self.sample_buffer()
                    # noinspection PyProtectedMember
                    if not is_eval_period:
                        new_o1 = self.env.preprocess_obs(
                            sample_steps.o1, shape=[self.batch_size, -1])
                        new_o2 = self.env.preprocess_obs(
                            sample_steps.o2, shape=[self.batch_size, -1])
                        step = self.agent.train_step(
                            sample_steps._replace(
                                o1=new_o1,
                                o2=new_o2,
                            ))
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
            s1 = s2
            episode_mean.update(Counter(fps=1 / float(time.time() - tick)))
            tick = time.time()
            episode_count.update(Counter(reward=r, time_steps=1))
            if t:
                s1 = self.reset()
                episode_reward = episode_count['reward']
                episode_time_steps = episode_count['time_steps']
                count.update(Counter(reward=episode_reward, episode=1))
                print('({}) Episode {}\t Time Steps: {}\t Reward: {}'.format(
                    'EVAL' if is_eval_period else 'TRAIN', count['episode'], time_steps,
                    episode_reward))
                if logdir:
                    summary = tf.Summary()
                    if is_eval_period:
                        summary.value.add(tag='eval reward', simple_value=episode_reward)
                    summary.value.add(
                        tag='average reward',
                        simple_value=(count['reward'] / float(count['episode'])))
                    for k in episode_count:
                        summary.value.add(tag=k, simple_value=episode_count[k])
                    for k in episode_mean:
                        summary.value.add(
                            tag=k,
                            simple_value=episode_mean[k] / float(episode_time_steps))
                    tb_writer.add_summary(summary, time_steps)
                    tb_writer.flush()

                # zero out counters
                self.episode_count = episode_count = Counter()
                episode_mean = Counter()

    def build_agent(self, base_agent: AbstractAgent = AbstractAgent, **kwargs):
        state_shape = self.env.observation_space.shape
        if isinstance(self.env.action_space, spaces.Discrete):
            action_shape = [self.env.action_space.n]
            policy_type = CategoricalPolicy
        else:
            action_shape = self.env.action_space.shape
            policy_type = GaussianPolicy

        class Agent(policy_type, base_agent):
            def __init__(self):
                super(Agent, self).__init__(o_shape=state_shape,
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

    def vectorize_state(self, state: Obs) -> np.ndarray:
        """ Preprocess state before feeding to network """
        return state

    def add_to_buffer(self, step: Step) -> None:
        assert isinstance(step, Step)
        self.buffer.append(step)

    def buffer_full(self):
        return len(self.buffer) >= self.batch_size

    def sample_buffer(self):
        indices = np.random.randint(
            -len(self.buffer), 0, size=self.batch_size)  # type: np.ndarray
        return Step(*self.buffer[indices])

    def trajectory(self, final_index: int = None) -> Optional[Step]:
        if final_index is None:
            final_index = 0  # points to current time step
        else:
            final_index -= self.time_steps()
        if self.buffer.empty:
            return None
        return self.buffer[-self.time_steps():final_index]

    def time_steps(self):
        return self.episode_count['time_steps']

    def _trajectory(self) -> Iterable:
        if self.time_steps():
            return self.buffer[-self.time_steps():]
        return ()


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

    def vectorize_state(self, state: Obs) -> np.ndarray:
        assert isinstance(self.env, HindsightWrapper)
        return self.env.preprocess_obs(state)
