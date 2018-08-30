import itertools
import time
from collections import Counter, namedtuple
from typing import Iterable, Optional, Tuple

import gym
import numpy as np
import tensorflow as tf
from gym import Wrapper, spaces

from environments.hindsight_wrapper import HindsightWrapper
from environments.multi_task import MultiTaskEnv
from sac.agent import AbstractAgent
from sac.policies import CategoricalPolicy, GaussianPolicy
from sac.replay_buffer import ReplayBuffer
from sac.utils import Obs, Step, create_sess, normalize, unwrap_env, vectorize

Agents = namedtuple('Agents', 'train act')


class Trainer:
    def __init__(self,
                 env: gym.Env,
                 seed: Optional[int],
                 buffer_size: int,
                 batch_size: int,
                 seq_len: int,
                 num_train_steps: int,
                 sess: tf.Session = None,
                 preprocess_func=None,
                 action_space=None,
                 observation_space=None,
                 **kwargs):

        obs = env.reset()

        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
            env.seed(seed)

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.env = env
        self.buffer = ReplayBuffer(buffer_size)
        self.sess = sess or create_sess()
        self.action_space = action_space or env.action_space
        self.observation_space = observation_space or env.observation_space

        self.agents = Agents(
            act=self.build_agent(
                sess=self.sess,
                batch_size=None,
                seq_len=1,
                reuse=False,
                action_space=action_space,
                observation_space=observation_space,
                **kwargs),
            train=None)
        # train=self.build_agent(
        #     sess=self.sess,
        #     batch_size=batch_size,
        #     seq_len=seq_len,
        #     reuse=True,
        #     action_space=action_space,
        #     observation_space=observation_space,
        #     **kwargs))
        self.seq_len = self.agents.act.seq_len
        self.count = Counter(reward=0, episode=0, time_steps=0)
        self.episode_count = Counter()

        self.preprocess_func = preprocess_func
        if preprocess_func is None and not isinstance(obs, np.ndarray):
            try:
                self.preprocess_func = unwrap_env(
                    env, lambda e: hasattr(e, 'preprocess_obs')).preprocess_obs
            except RuntimeError:
                self.preprocess_func = vectorize

        # self.train(load_path, logdir, render, save_path)

    def train(self, load_path, logdir, render, save_path):
        saver = tf.train.Saver()
        tb_writer = None
        if load_path:
            saver.restore(self.sess, load_path)
            print("Model restored from", load_path)
        if logdir:
            tb_writer = tf.summary.FileWriter(logdir=logdir, graph=self.sess.graph)

        for episodes in itertools.count(1):
            if save_path and episodes % 25 == 1:
                print("model saved in path:", saver.save(self.sess, save_path=save_path))
                saver.save(self.sess, save_path.replace('<episode>', str(episodes)))
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
                        summary.value.add(tag=k.replace('_', ' '),
                                          simple_value=self.episode_count[k])
                tb_writer.add_summary(summary, self.count['time_steps'])
                tb_writer.flush()

    def is_eval_period(self):
        return self.count['episode'] % 100 == 99

    def trajectory(self, final_index=None) -> Optional[Step]:
        if final_index is None:
            final_index = 0  # points to current time step
        else:
            final_index -= self.time_steps()  # relative to start of episode
        if self.buffer.empty:
            return None
        return Step(*self.buffer[-self.time_steps():final_index])

    def time_steps(self):
        return self.episode_count['time_steps']

    def run_episode(self, o1, perform_updates, render):
        episode_count = Counter()
        episode_mean = Counter()
        tick = time.time()
        s = self.agents.act.initial_state
        for time_steps in itertools.count(1):
            a, s = self.get_actions(o1, s)
            o2, r, t, info = self.step(a, render)
            if 'print' in info:
                print('Time step:', time_steps, info['print'])
            if 'log count' in info:
                episode_count.update(Counter(info['log count']))
            if 'log mean' in info:
                episode_mean.update(Counter(info['log mean']))
            self.add_to_buffer(Step(o1=o1, a=a, r=r, o2=o2, t=t))

            if self.buffer_full() and perform_updates:
                for i in range(self.num_train_steps):
                    step = self.agents.act.train_step(self.sample_buffer())
                    episode_mean.update(
                        Counter({
                            k.replace(' ', '_'): v
                            for k, v in step.items() if np.isscalar(v)
                        }))
            o1 = o2
            episode_mean.update(Counter(fps=1 / float(time.time() - tick)))
            tick = time.time()
            episode_count.update(Counter(reward=r, time_steps=1))
            if t:
                for k in episode_mean:
                    episode_count[k] = episode_mean[k] / float(time_steps)
                return episode_count

    def perform_update(self):
        counter = Counter()
        if self.buffer_full():
            for i in range(self.num_train_steps):
                step = self.agents.act.train_step(self.sample_buffer())
                counter.update(
                    Counter({
                        k.replace(' ', '_'): v
                        for k, v in step.items() if np.isscalar(v)
                    }))
        return counter

    def get_actions(self, o1, s):
        return self.agents.act.get_actions(
            self.preprocess_obs(o1), state=s, sample=(not self.is_eval_period()))

    def build_agent(self,
                    base_agent: AbstractAgent,
                    action_space=None,
                    observation_space=None,
                    **kwargs):
        if observation_space is None:
            observation_space = self.observation_space
        if action_space is None:
            action_space = self.action_space
        if isinstance(action_space, spaces.Discrete):
            action_shape = [action_space.n]
            policy_type = CategoricalPolicy
        else:
            action_shape = action_space.shape
            policy_type = GaussianPolicy

        if isinstance(observation_space, spaces.Discrete):
            state_shape = [observation_space.n]
        else:
            state_shape = observation_space.shape

        class Agent(policy_type, base_agent):
            def __init__(self):
                super(Agent, self).__init__(
                    o_shape=state_shape, a_shape=action_shape, **kwargs)

        return Agent()

    def reset(self) -> Obs:
        return self.env.reset()

    def step(self, action: np.ndarray, render: bool) -> Tuple[Obs, float, bool, dict]:
        """ Preprocess action before feeding to env """
        if render:
            self.env.render()
        if type(self.action_space) is spaces.Discrete:
            # noinspection PyTypeChecker
            return self.env.step(np.argmax(action))
        else:
            action = np.tanh(action)
            hi, lo = self.action_space.high, self.action_space.low
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

    def sample_buffer(self) -> Step:
        sample = Step(*self.buffer.sample(self.batch_size, seq_len=self.seq_len))
        if self.seq_len is None:
            # leave state as dummy value for non-recurrent
            shape = [self.batch_size, -1]
            return Step(
                o1=self.preprocess_obs(sample.o1, shape=[self.batch_size, -1]),
                o2=self.preprocess_obs(sample.o2, shape=[self.batch_size, -1]),
                # s=sample.s,
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

    def preprocess_obs(self, obs, shape: tuple = None):
        if self.preprocess_func is not None:
            obs = self.preprocess_func(obs, shape)
        return obs
        # return normalize(
        #     vector=obs,
        #     low=self.env.observation_space.low,
        #     high=self.env.observation_space.high)


class TrajectoryTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_to_buffer(self, step: Step):
        super().add_to_buffer(step)

    def trajectory(self) -> Iterable:
        return self.buffer[-self.episode_count['time_steps']:]

    def reset(self) -> Obs:
        return super().reset()

    def time_steps(self):
        return self.episode_count['time_steps']

    def _trajectory(self) -> Iterable:
        if self.time_steps():
            return self.buffer[-self.time_steps():]
        return ()


class HindsightTrainer(TrajectoryTrainer):
    def __init__(self, env: Wrapper, n_goals: int, **kwargs):
        self.n_goals = n_goals
        self.hindsight_env = env
        while not isinstance(self.hindsight_env, HindsightWrapper):
            try:
                self.hindsight_env = self.hindsight_env.env
            except AttributeError:
                raise RuntimeError(f"env {env} must include HindsightWrapper.")
        assert isinstance(self.hindsight_env, HindsightWrapper)
        super().__init__(env=env, **kwargs)

    def add_hindsight_trajectories(self) -> None:
        assert isinstance(self.hindsight_env, HindsightWrapper)
        return
        self.buffer.extend(
            self.hindsight_env.recompute_trajectory(
                self._trajectory(), final_step=self.buffer[-1]))
        if self.n_goals - 1 and self.time_steps() > 0:
            final_indexes = np.random.randint(
                1, self.time_steps(), size=self.n_goals - 1) - self.time_steps()
            assert isinstance(final_indexes, np.ndarray)

            for final_state in self.buffer[final_indexes]:
                self.buffer.extend(
                    self.hindsight_env.recompute_trajectory(
                        self._trajectory(), final_step=final_state))

    def reset(self) -> Obs:
        self.add_hindsight_trajectories()
        return super().reset()


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
