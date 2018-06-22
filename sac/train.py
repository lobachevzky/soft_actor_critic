import itertools
import time
from collections import Counter, namedtuple
from pprint import pprint
from typing import Iterable, Optional, Tuple, List

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
from sac.utils import State, Step, ArrayLike

Agents = namedtuple('Agents', 'train act')
Buffers = namedtuple('Buffers', 'o a r t')


class Trainer:
    def __init__(self, base_agent: AbstractAgent, env: gym.Env, seed: Optional[int],
                 buffer_size: int, batch_size: int, num_train_steps: int,
                 logdir: str, save_path: str, load_path: str, render: bool, **kwargs):

        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
            env.seed(seed)

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.env = env
        self.save_path = save_path

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.inter_op_parallelism_threads = 1
        sess = tf.Session(config=config)

        self.agents = Agents(act=self.build_agent(sess=sess,
                                                  base_agent=base_agent,
                                                  batch_size=1,
                                                  reuse=False,
                                                  **kwargs),
                             train=self.build_agent(sess=sess,
                                                    base_agent=base_agent,
                                                    batch_size=batch_size,
                                                    reuse=True,
                                                    **kwargs))
        saver = tf.train.Saver()

        self.buffer = self.build_buffer(maxlen=buffer_size,
                                        obs=env.reset(),
                                        action=env.action_space.sample(),
                                        initial_state=base_agent.initial_state)

        tb_writer = None
        if load_path:
            saver.restore(sess, load_path)
            print("Model restored from", load_path)
        if logdir:
            tb_writer = tf.summary.FileWriter(logdir=logdir,
                                              graph=sess.graph)

        self.count = count = Counter(reward=0, episode=0, time_steps=0)
        self.episode_count = Counter()

        for episodes in itertools.count(1):
            if save_path and episodes % 25 == 1:
                print("model saved in path:", saver.save(sess, save_path=save_path))
                saver.save(sess, save_path.replace('<episode>', str(episodes)))
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
        assert isinstance(self.agents.act, AbstractAgent)
        assert isinstance(self.agents.train, AbstractAgent)
        episode_count = Counter()
        episode_mean = Counter()
        tick = time.time()
        s = self.agents.act.initial_state
        for time_steps in itertools.count(1):
            a, s = self.agents.act.get_actions(
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
                    step = self.agents.train.train_step(self.sample_buffer())
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

    def build_agent(self, base_agent: AbstractAgent, batch_size, reuse, **kwargs):
        state_shape = self.env.observation_space.shape
        if isinstance(self.env.action_space, spaces.Discrete):
            action_shape = [self.env.action_space.n]
            policy_type = CategoricalPolicy
        else:
            action_shape = self.env.action_space.shape
            policy_type = GaussianPolicy

        class Agent(policy_type, base_agent):
            def __init__(self, batch_size, reuse):
                super(Agent, self).__init__(batch_size=batch_size,
                                            o_shape=state_shape,
                                            a_shape=action_shape,
                                            reuse=reuse,
                                            **kwargs)

        return Agent(batch_size=batch_size, reuse=reuse)

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

    @staticmethod
    def build_buffer(maxlen, obs: ArrayLike, action: ArrayLike,
                     initial_state: ArrayLike):
        def dim(array: ArrayLike):
            shape = np.shape(array)  # type: Tuple[int]
            return shape[-1]

        return ReplayBuffer(maxlen, Buffers(o=dim(obs),
                                            a=dim(action),
                                            r=1,
                                            t=1))

    def add_to_buffer(self, step: Step) -> None:
        assert isinstance(step, Step)
        self.buffer.append(step)

    def buffer_full(self):
        return len(self.buffer) >= self.batch_size

    def sample_buffer(self) -> Step:
        buffer_sample = Buffers(self.buffer.sample(self.batch_size))
        sample = Step(*(buffer_sample))

        def last(x: List[list]):
            return [y[-1] for y in x]

        sample = Step(o1=self.vectorize_state(sample.o1),
                      o2=self.vectorize_state(sample.o2),
                      s=last(sample.s),
                      a=last(sample.a),
                      r=last(sample.r),
                      t=last(sample.t))
        return sample


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

HindsightBuffers = namedtuple('HindsightBuffers', 'o ag dg a r t')

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
