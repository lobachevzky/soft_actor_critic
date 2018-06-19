import itertools
import time
from collections import Counter
from typing import Iterable, Optional, Tuple

import gym
import numpy as np
import tensorflow as tf
from gym import spaces

from environments.hindsight_wrapper import HindsightWrapper
from environments.multi_task import MultiTaskEnv
from sac.agent import AbstractAgent
from sac.networks import LstmAgent
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

        self.count = count = Counter(reward=0, episode=0, time_steps=0)
        self.episode_count = Counter()

        for episodes in itertools.count(1):
            if save_path and episodes % 25 == 1:
                _save_path = save_path.replace('<episode>', str(episodes))
                print("model saved in path:", saver.save(
                    agent.sess, save_path=_save_path))
            self.episode_count = self.run_episode(
                o1=self.reset(),
                render=render,
                perform_updates=not self.is_eval_period() and load_path is None)
            episode_reward = self.episode_count['reward']
            episode_timesteps = self.episode_count['timesteps']
            count.update(
                Counter(reward=episode_reward, episode=1, time_steps=episode_timesteps))
            print('({}) Episode {}\t Time Steps: {}\t Reward: {}'.format(
                'EVAL' if self.is_eval_period() else 'TRAIN', episodes,
                count['time_steps'], episode_reward))
            if logdir:
                summary = tf.Summary()
                if self.is_eval_period():
                    summary.value.add(tag='eval reward', simple_value=episode_reward)
                else:
                    for k in self.episode_count:
                        summary.value.add(tag=k, simple_value=self.episode_count[k])
                tb_writer.add_summary(summary, count['time_steps'])
                tb_writer.flush()

    def is_eval_period(self):
        return self.count['episode'] % 100 == 99

    def run_episode(self, o1, perform_updates, render):
        assert isinstance(self.agent, AbstractAgent)
        episode_count = Counter()
        episode_mean = Counter()
        tick = time.time()
        s = self.agent.initial_state
        for time_steps in itertools.count(1):
            a, s = self.agent.get_actions(
                self.vectorize_state(o1), s, sample=(not self.is_eval_period()))
            if render:
                self.env.render()
            o2, r, t, info = self.step(a)
            if 'print' in info:
                print('Time step:', time_steps, info['print'])
            if 'log count' in info:
                episode_count.update(Counter(info['log count']))
            if 'log mean' in info:
                episode_mean.update(Counter(info['log mean']))
            self.add_to_buffer(Step(s=s, o1=o1, a=a, r=r, o2=o2, t=t))

            if self.buffer_full() and perform_updates:
                for i in range(self.num_train_steps):
                    sample_steps = self.sample_buffer()
                    step = self.agent.train_step(
                        sample_steps.replace(
                            o1=self.vectorize_state(sample_steps.o1),
                            o2=self.vectorize_state(sample_steps.o2),
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
            o1 = o2
            episode_mean.update(Counter(fps=1 / float(time.time() - tick)))
            tick = time.time()
            episode_count.update(Counter(reward=r, timesteps=1))
            if t:
                for k in episode_mean:
                    episode_count[k] = episode_mean[k] / float(time_steps)
                return episode_count

    def build_agent(self, base_agent: AbstractAgent, **kwargs):
        state_shape = self.env.observation_space.shape
        if isinstance(self.env.action_space, spaces.Discrete):
            action_shape = [self.env.action_space.n]
            policy_type = CategoricalPolicy
        else:
            action_shape = self.env.action_space.shape
            policy_type = GaussianPolicy

        class Agent(policy_type, base_agent):
            def __init__(self, s_shape, a_shape):
                super(Agent, self).__init__(o_shape=s_shape, a_shape=a_shape, **kwargs)

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


class MemoryTrainer(Trainer):
    def __init__(self, seq_length: int, env: gym.Env, seed: Optional[int], buffer_size: int, batch_size: int,
                 num_train_steps: int,
                 logdir: str, save_path: str, load_path: str, render: bool, **kwargs):
        self.seq_length = seq_length
        super().__init__(env, seed, buffer_size, batch_size, num_train_steps, logdir, save_path, load_path, render,
                         **kwargs)

    def sample_buffer(self):
        return Step(*self.buffer.sample(self.batch_size, self.seq_length))


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
            final_indexes = np.random.randint(1, self.timesteps(), size=self.n_goals - 1) - self.timesteps()
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
    def __init__(self, evaluation, env: HindsightWrapper, n_goals: int, **kwargs):
        self.eval = evaluation
        super().__init__(env, n_goals, **kwargs)

    def run_episode(self, o1, perform_updates, render):
        if not self.is_eval_period():
            return super().run_episode(
                o1=o1, perform_updates=perform_updates, render=render)
        env = self.env.unwrapped
        assert isinstance(env, MultiTaskEnv), type(env)
        all_goals = itertools.product(*env.goals)
        for goal in all_goals:
            o1 = self.reset()
            env.set_goal(goal)
            count = super().run_episode(
                o1=o1, perform_updates=perform_updates, render=render)
            for k in count:
                print(f'{k}: {count[k]}')
        print('Evaluation complete.')
        exit()

    def is_eval_period(self):
        return self.eval
