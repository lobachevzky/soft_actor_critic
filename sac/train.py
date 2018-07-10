import itertools
import pickle
import time
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import Optional, Tuple, Iterable, Any

import gym
import numpy as np
import tensorflow as tf
from gym import spaces

from environments.old_hindsight_wrapper import HindsightWrapper
from sac.agent import AbstractAgent
from sac.networks import MlpAgent
from sac.policies import CategoricalPolicy, GaussianPolicy
from sac.replay_buffer import ReplayBuffer
from sac.utils import Step
State = Any


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
        self.save_path = save_path

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.inter_op_parallelism_threads = 1
        sess = tf.Session(config=config)

        self.agent = agent = self.build_agent(
                sess=sess,
                base_agent=MlpAgent,
                batch_size=None,
                seq_len=1,
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
            self.add_to_buffer(Step(s=None, o1=s1, a=a, r=r, o2=s2, t=t))
            if self.buffer_full() and not load_path:
                for i in range(self.num_train_steps):
                    sample_steps = self.sample_buffer()
                    # noinspection PyProtectedMember
                    if not is_eval_period:
                        step = self.agent.train_step(
                            sample_steps._replace(
                                o1=list(map(self.vectorize_state, sample_steps.o1)),
                                o2=list(map(self.vectorize_state, sample_steps.o2)),
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
        return Step(*self.buffer.sample(self.batch_size))


class TrajectoryTrainer(Trainer):
    def __init__(self, mimic_save_dir: Optional[str], **kwargs):
        self.mimic_save_dir = mimic_save_dir
        if mimic_save_dir is not None:
            path = Path(mimic_save_dir)
            print('Using model dir', path.absolute())
            path.mkdir(parents=True, exist_ok=True)
        self.mimic_num = 0
        self.trajectory = []
        self.stem_num = 0
        super().__init__(**kwargs)

    def add_to_buffer(self, step: Step):
        super().add_to_buffer(step)
        self.trajectory.append(step)

    def trajectory(self) -> Iterable:
        assert len(self.trajectory) == self.episode_count['timesteps']
        return self.buffer[-self.episode_count['timesteps']:]

    def reset(self) -> State:
        if self.mimic_save_dir is not None:
            path = Path(self.mimic_save_dir, str(self.mimic_num) + '.pkl')
            with path.open(mode='wb') as f:
                pickle.dump(self.trajectory, f)
            self.mimic_num += 1
        self.trajectory = []
        return super().reset()

    def timesteps(self):
        return self.episode_count['timesteps']

    def _trajectory(self) -> Iterable:
        if self.timesteps():
            return self.buffer[-self.timesteps():]
        return ()


def steps_are_same(step1, step2):
    return True
    if step1 is None or step2 is None:
        return False
    return all([
        np.allclose(step1.o1.obs, step2.o1.obs),
        np.allclose(step1.o2.obs, step2.o2.obs)])


class HindsightTrainer(TrajectoryTrainer):
    def __init__(self, env: HindsightWrapper, n_goals: int, **kwargs):
        self.n_goals = n_goals
        assert isinstance(env, HindsightWrapper)
        super().__init__(env=env, **kwargs)

    def add_hindsight_trajectories(self) -> None:
        for i, (step1, step2) in enumerate(zip(self.trajectory, self._trajectory())):
            if step1 is None:
                print('step', i, 'in self.trajectory is None')
            if step2 is None:
                print('step', i, 'in buffer slice is None')
            if not steps_are_same(step2, self.buffer[-self.timesteps() + i]):
                print('issue with indexing')
                print('step in buffer slice:')
                pprint(step2)
                print('step in buffer index (self.buffer[{}])'.format(i))
                pprint(self.buffer[i])
                exit()

            if not steps_are_same(step1, step2):
                print('encountered difference at step', i)
                print('step in self.trajectory:')
                pprint(step1)
                print('step in buffer slice:')
                pprint(step2)
                print('same as i - 1', steps_are_same(self.buffer[i - 1], step1))
                print('same as i', steps_are_same(self.buffer[i], step1))
                print('same as i + 1', steps_are_same(self.buffer[i + 1], step1))
                exit()

        assert isinstance(self.env, HindsightWrapper)
        if self.trajectory:
            assert steps_are_same(self.buffer[-1], self.trajectory[-1])
        self.buffer.extend(self.env.recompute_trajectory(self._trajectory()))
        if self.n_goals - 1 and self.timesteps() > 0:
            final_indexes = np.random.randint(1, self.timesteps(), size=self.n_goals - 1)
            assert isinstance(final_indexes, np.ndarray)

            for final_state in self.buffer[final_indexes]:
                self.buffer.extend(self.env.recompute_trajectory(self._trajectory()))

    def reset(self) -> State:
        self.add_hindsight_trajectories()
        return super().reset()

    def vectorize_state(self, state: State) -> np.ndarray:
        assert isinstance(self.env, HindsightWrapper)
        return self.env.vectorize_state(state)
