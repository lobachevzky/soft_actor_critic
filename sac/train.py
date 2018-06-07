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

from environments.hindsight_wrapper import HindsightWrapper
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

        self.logdir = logdir
        self.save_path = save_path
        self.load_path = load_path
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

        self.agent = self.build_agent(**kwargs)

        self.saver = tf.train.Saver()
        self.tb_writer = None
        if load_path:
            self.saver.restore(self.agent.sess, load_path)
            print("Model restored from", load_path)
        if logdir:
            self.tb_writer = tf.summary.FileWriter(logdir=logdir, graph=self.agent.sess.graph)

        self.count = Counter(reward=0, episode=0)
        self.episode_count = Counter()
        self.episode_mean = Counter()

    def train(self):
        tick = time.time()
        s1 = self.reset()
        for time_steps in itertools.count():
            is_eval_period = self.count['episode'] % 100 == 99
            a = self.agent.get_actions(self.vectorize_state(s1), sample=(not is_eval_period))
            if self.render:
                self.env.render()
            s2, r, t, info = self.step(a)
            if 'print' in info:
                print('Time step:', time_steps, info['print'])
            if 'log count' in info:
                self.episode_count.update(Counter(info['log count']))
            if 'log mean' in info:
                self.episode_mean.update(Counter(info['log mean']))

            if self.save_path and time_steps % 5000 == 0:
                print("model saved in path:", self.saver.save(self.agent.sess,
                                                              save_path=self.save_path))
            self.add_to_buffer(Step(s1=s1, a=a, r=r, s2=s2, t=t))
            if not is_eval_period and self.buffer_full() and not self.load_path:
                for i in range(self.num_train_steps):
                    sample_steps = self.sample_buffer()
                    step = self.agent.train_step(
                        sample_steps.replace(
                            s1=self.vectorize_state(sample_steps.s1),
                            s2=self.vectorize_state(sample_steps.s2),
                        ))
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
            s1 = s2
            self.episode_mean.update(Counter(fps=1 / float(time.time() - tick)))
            tick = time.time()
            self.episode_count.update(Counter(reward=r, timesteps=1))
            if t:
                s1 = self.reset()
                episode_reward = self.episode_count['reward']
                episode_timesteps = self.episode_count['timesteps']
                self.count.update(Counter(reward=episode_reward, episode=1))
                print('({}) Episode {}\t Time Steps: {}\t Reward: {}'.format(
                    'EVAL' if is_eval_period else 'TRAIN', self.count['episode'], time_steps,
                    episode_reward))
                if self.logdir:
                    summary = tf.Summary()
                    if is_eval_period:
                        summary.value.add(tag='eval reward', simple_value=episode_reward)
                    summary.value.add(
                        tag='average reward',
                        simple_value=(self.count['reward'] / float(self.count['episode'])))
                    for k in self.episode_count:
                        summary.value.add(tag=k, simple_value=self.episode_count[k])
                    for k in self.episode_mean:
                        summary.value.add(
                            tag=k,
                            simple_value=self.episode_mean[k] / float(episode_timesteps))
                    self.tb_writer.add_summary(summary, time_steps)
                    self.tb_writer.flush()

                # zero out counters
                self.episode_count = Counter()
                self.episode_mean = Counter()

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
    def __init__(self, mimic_save_dir: Optional[str], **kwargs):
        self.mimic_save_dir = mimic_save_dir
        if mimic_save_dir is not None:
            path = Path(mimic_save_dir)
            print('Using model dir', path.absolute())
            path.mkdir(parents=True, exist_ok=True)
        self.mimic_num = 0
        super().__init__(**kwargs)

    def add_to_buffer(self, step: Step):
        super().add_to_buffer(step)

    def trajectory(self) -> Iterable:
        return self.buffer[-self.episode_count['timesteps']:]

    def reset(self) -> State:
        if self.mimic_save_dir is not None:
            path = Path(self.mimic_save_dir, str(self.mimic_num) + '.pkl')
            with path.open(mode='wb') as f:
                pickle.dump(self._trajectory(), f)
            self.mimic_num += 1
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
