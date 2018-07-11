import itertools
import time
from collections import Counter
from typing import Optional, Tuple, Iterable, Any

import gym
import numpy as np
import tensorflow as tf
from gym import spaces

import sac.old_replay_buffer
import sac.replay_buffer
from environments.hindsight_wrapper import Observation
from environments.old_hindsight_wrapper import HindsightWrapper
from sac.agent import AbstractAgent
from sac.policies import CategoricalPolicy, GaussianPolicy
from sac.utils import Step

State = Any


class Trainer:
    def __init__(self, env: gym.Env, seed: Optional[int], buffer_size: int,
                 batch_size: int, num_train_steps: int, logdir: str, save_path: str, load_path: str, render: bool,
                 **kwargs):

        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
            env.seed(seed)

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.env = env
        self.buffer = sac.replay_buffer.ReplayBuffer(buffer_size)
        self.save_path = save_path

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.inter_op_parallelism_threads = 1
        sess = tf.Session(config=config)

        self.agent = agent = self.build_agent(
            sess=sess,
            batch_size=None,
            reuse=False,
            **kwargs)
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
            a = agent.get_actions([self.vectorize_state(s1)], sample=(not is_eval_period)).output
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
                        new_o1 = self.env.preprocess_obs(sample_steps.o1, shape=[self.batch_size, -1])
                        new_o2 = self.env.preprocess_obs(sample_steps.o2, shape=[self.batch_size, -1])
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
            episode_count.update(Counter(reward=r, timesteps=1))
            if t:
                s1 = self.reset()
                episode_reward = episode_count['reward']
                episode_timesteps = episode_count['timesteps']
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
                            simple_value=episode_mean[k] / float(episode_timesteps))
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
            def __init__(self, s_shape, a_shape):
                super(Agent, self).__init__(
                    o_shape=s_shape,
                    a_shape=a_shape,
                    **kwargs)

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

    def sample_buffer(self):
        indices = np.random.randint(-len(self.buffer), 0, size=self.batch_size)  # type: np.ndarray
        return Step(*self.buffer[indices])


class TrajectoryTrainer(Trainer):
    def __init__(self, **kwargs):
        self.mimic_num = 0
        self.stem_num = 0
        super().__init__(**kwargs)

    def trajectory(self, final_index=None) -> Optional[Step]:
        if final_index is None:
            final_index = 0  # points to current time step
        else:
            final_index -= self.timesteps()
        if self.buffer.empty:
            return None
        return self.buffer[-self.timesteps():final_index]

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
        if self.trajectory():
            new_trajectory = self.trajectory()
            old_trajectory = list(self._trajectory())

            if self.timesteps() > 0:
                new_recomputed_trajectory = self.env.recompute_trajectory(new_trajectory)
                assert Step(*new_recomputed_trajectory[-1]).t == True
                self.buffer.append(self.env.recompute_trajectory(new_trajectory))
                assert Step(*self.buffer[-1]).t == True
        # if self.n_goals - 1 and self.timesteps() > 0:
        #     final_indexes = np.random.randint(1, self.timesteps(), size=self.n_goals - 1)
        #     assert isinstance(final_indexes, np.ndarray)
        #
        #     for final_state in self.old_buffer[final_indexes]:
        #         self.old_buffer.extend(self.env.old_recompute_trajectory(self._trajectory()))

    def reset(self) -> State:
        self.add_hindsight_trajectories()
        return super().reset()

    def vectorize_state(self, state: State) -> np.ndarray:
        assert isinstance(self.env, HindsightWrapper)
        return self.env.old_vectorize_state(state)
