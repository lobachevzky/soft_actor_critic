import itertools
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Tuple

import gym
import numpy as np
import tensorflow as tf
from gym import spaces

from environments.hindsight_wrapper import HindsightWrapper, MultiTaskHindsightWrapper
from sac.agent import AbstractAgent
from sac.policies import CategoricalPolicy, GaussianPolicy
from sac.replay_buffer import ReplayBuffer
from sac.utils import State, Step


class Trainer:
    def __init__(self, env: gym.Env, seed: Optional[int], buffer_size: int,
                 batch_size: int, num_train_steps: int, logdir: str, save_path: str,
                 load_path: str, render: bool, **kwargs):

        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
            env.seed(seed)

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.env = env
        self.buffer = ReplayBuffer(buffer_size)
        self.save_path = save_path
        self.agent = agent = self.build_agent(**kwargs)

        saver = tf.train.Saver()
        tb_writer = None
        if load_path:
            saver.restore(agent.sess, load_path)
            print("Model restored from", load_path)
        if logdir:
            tb_writer = tf.summary.FileWriter(logdir=logdir, graph=agent.sess.graph)

        count = Counter(reward=0, episode=0, time_steps=0)
        self.episode_count = episode_count = Counter()
        self.episode_mean = episode_mean = Counter()

        s1 = self.reset()

        for episodes in itertools.count():
            if save_path and episodes % 25 == 0:
                print("model saved in path:", saver.save(agent.sess, save_path=save_path))
            is_eval_period = count['episode'] % 100 == 99
            self.run_episode(agent, count, env, episode_count, episode_mean, load_path, render, s1,
                             save_path, saver)
            s1 = self.reset()
            episode_reward = episode_count['reward']
            episode_timesteps = episode_count['timesteps']
            count.update(Counter(reward=episode_reward, episode=1, time_steps=episode_timesteps))
            print('({}) Episode {}\t Time Steps: {}\t Reward: {}'.format(
                'EVAL' if is_eval_period else 'TRAIN', count['episode'], count['time_steps'],
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
                        simple_value=episode_mean[k] / float(episode_timesteps))
                tb_writer.add_summary(summary, count['time_steps'])
                tb_writer.flush()

            # zero out counters
            self.episode_count = episode_count = Counter()
            episode_mean = Counter()

    def run_episode(self, agent, count, env, episode_count, episode_mean, load_path, render, s1, save_path, saver):
        tick = time.time()
        for time_steps in itertools.count():
            is_eval_period = count['episode'] % 100 == 99
            a = agent.get_actions(self.vectorize_state(s1), sample=(not is_eval_period))
            if render:
                env.render()
            s2, r, t, info = self.step(a)
            if 'print' in info:
                print('Time step:', time_steps, info['print'])
            if 'log count' in info:
                episode_count.update(Counter(info['log count']))
            if 'log mean' in info:
                episode_mean.update(Counter(info['log mean']))
            self.add_to_buffer(Step(s1=s1, a=a, r=r, s2=s2, t=t))

            if not is_eval_period and self.buffer_full() and not load_path:
                for i in range(self.num_train_steps):
                    sample_steps = self.sample_buffer()
                    step = self.agent.train_step(
                        sample_steps.replace(
                            s1=self.vectorize_state(sample_steps.s1),
                            s2=self.vectorize_state(sample_steps.s2),
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
            episode_count.update(Counter(reward=r, timesteps=1))
            if t:
                return

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
                super(Agent, self).__init__(s_shape=s_shape, a_shape=a_shape, **kwargs)

        return Agent(state_shape, action_shape)

    def reset(self) -> State:
        return self.env.reset()

    def step(self, action: np.ndarray) -> Tuple[State, float, bool, dict]:
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

    def sample_buffer(self) -> Step:
        return Step(*self.buffer.sample(self.batch_size))


class TrajectoryTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_to_buffer(self, step: Step):
        super().add_to_buffer(step)

    def trajectory(self) -> Iterable:
        return self.buffer[-self.episode_count['timesteps']:]

    def reset(self) -> State:
        return super().reset()

    def timesteps(self):
        return self.episode_count['timesteps']

    def _trajectory(self) -> Iterable:
        if self.timesteps():
            return self.buffer[-self.timesteps():]
        return ()


class HindsightTrainer(TrajectoryTrainer):
    def __init__(self, env: HindsightWrapper, n_goals: int, **kwargs):
        self.n_goals = n_goals
        assert isinstance(env, HindsightWrapper)
        super().__init__(env=env, **kwargs)

    def add_hindsight_trajectories(self) -> None:
        assert isinstance(self.env, HindsightWrapper)
        self.buffer.extend(
            self.env.recompute_trajectory(self._trajectory(), final_step=self.buffer[-1]))
        if self.n_goals - 1 and self.timesteps() > 0:
            final_indexes = np.random.randint(1, self.timesteps(), size=self.n_goals - 1)
            assert isinstance(final_indexes, np.ndarray)

            for final_state in self.buffer[final_indexes]:
                self.buffer.extend(
                    self.env.recompute_trajectory(
                        self._trajectory(), final_step=final_state))

    def reset(self) -> State:
        self.add_hindsight_trajectories()
        return super().reset()

    def vectorize_state(self, state: State) -> np.ndarray:
        assert isinstance(self.env, HindsightWrapper)
        return self.env.vectorize_state(state)


class MultiTaskHindsightTrainer(HindsightTrainer):
    def _trajectory(self):
        if self.timesteps():
            steps = list(self.buffer[-(self.timesteps() + 1):-1])
            assert len(steps) == self.timesteps()
            return steps
        return ()

