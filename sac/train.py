import itertools
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Tuple

import gym
import numpy as np
import tensorflow as tf
from gym import spaces

from environments.hindsight_wrapper import HindsightWrapper
from environments.unsupervised import UnsupervisedEnv
from sac.agent import AbstractAgent, PropagationAgent
from sac.policies import CategoricalPolicy, GaussianPolicy
from sac.replay_buffer import ReplayBuffer
from sac.utils import PropStep, State, Step


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

        if mimic_dir:
            for path in Path(mimic_dir).iterdir():
                if path.suffix == '.pkl':
                    with Path(path).open('rb') as f:
                        self.buffer.extend(pickle.load(f))
                print('Loaded mimic file {} into buffer.'.format(path))

        s1 = self.reset()

        self.agent = agent = self.build_agent(**kwargs)

        if isinstance(env.unwrapped, UnsupervisedEnv):
            # noinspection PyUnresolvedReferences
            env.unwrapped.initialize(agent.sess, self.buffer)

        saver = tf.train.Saver()
        tb_writer = None
        if load_path:
            saver.restore(agent.sess, load_path)
            print("Model restored from", load_path)
        if logdir:
            tb_writer = tf.summary.FileWriter(logdir=logdir, graph=agent.sess.graph)

        count = Counter(reward=0, episode=0)
        episode_count = Counter()
        episode_mean = Counter()
        evaluation_period = 10

        for time_steps in itertools.count():
            is_eval_period = count['episode'] % evaluation_period == evaluation_period - 1
            a = agent.get_actions([self.vectorize_state(s1)], sample=(not is_eval_period))
            if render:
                env.render()
            s2, r, t, info = self.step(a)
            if 'print' in info:
                print('Time step:', time_steps, info['print'])
            if 'log count' in info:
                episode_count.update(Counter(info['log count']))
            if 'log mean' in info:
                episode_mean.update(Counter(info['log mean']))

            tick = time.time()
            if save_path and time_steps % 5000 == 0:
                print("model saved in path:", saver.save(agent.sess, save_path=save_path))
            if not is_eval_period:
                self.add_to_buffer(Step(s1=s1, a=a, r=r, s2=s2, t=t))
                if self.buffer_full() and not load_path:
                    for i in range(self.num_train_steps):
                        sample_steps = self.sample_buffer()
                        # noinspection PyProtectedMember
                        step = self.agent.train_step(
                            sample_steps._replace(
                                s1=list(map(self.vectorize_state, sample_steps.s1)),
                                s2=list(map(self.vectorize_state, sample_steps.s2)),
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
            episode_mean.update(Counter(fps=1 / float(time.time() - tick) + 1e-10))
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
                    tb_writer.add_summary(summary, count['episode'])
                    tb_writer.flush()

                # zero out counters
                episode_count = Counter()

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
                    s_shape=s_shape,
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
        super().__init__(**kwargs)
        self.s1 = self.reset()

    def step(self, action: np.ndarray) -> Tuple[State, float, bool, dict]:
        s2, r, t, i = super().step(action)
        self.trajectory.append(Step(s1=self.s1, a=action, r=r, s2=s2, t=t))
        self.s1 = s2
        return s2, r, t, i

    def reset(self) -> State:
        if self.mimic_save_dir is not None:
            path = Path(self.mimic_save_dir, str(self.mimic_num) + '.pkl')
            with path.open(mode='wb') as f:
                pickle.dump(self.trajectory, f)
            self.mimic_num += 1
        self.trajectory = []
        self.s1 = super().reset()
        return self.s1


class DoubleBufferTrainer(TrajectoryTrainer):
    def __init__(self, buffer_size: int, **kwargs):
        self.success = False
        self.success_buffer = ReplayBuffer(buffer_size)
        self.buffer = ReplayBuffer(buffer_size)
        super().__init__(buffer_size=buffer_size, **kwargs)

    def add_to_buffer(self, _):
        pass

    def step(self, action: np.ndarray):
        s, r, t, i = super().step(action)
        if r == 1:
            self.success = False
        return s, r, t, i

    def reset(self):
        if self.success:
            self.success_buffer.extend(self.trajectory)
        else:
            self.buffer.extend(self.trajectory)
        return super().reset()

    def buffer_full(self):
        return self.batch_size <= max(len(self.success_buffer), len(self.buffer))

    def sample_buffer(self):
        if self.batch_size <= min(len(self.success_buffer), len(self.buffer)):
            # both buffers have enough elements
            half_batch = self.batch_size // 2
            success_sample = self.success_buffer.sample(half_batch)
            failure_sample = self.buffer.sample(self.batch_size - half_batch)
            return Step(*map(np.concatenate, zip(success_sample, failure_sample)))
        for buffer in [self.success_buffer, self.buffer]:
            if self.batch_size <= len(buffer):
                # sample first buffer that has enough elements
                return Step(*buffer.sample(self.batch_size))
        raise RuntimeError(
            "This should be impossible. At least one buffer should be full.")


class HindsightTrainer(TrajectoryTrainer):
    def __init__(self, env: HindsightWrapper, n_goals: int, **kwargs):
        self.n_goals = n_goals
        assert isinstance(env, HindsightWrapper)
        super().__init__(env=env, **kwargs)

    def add_hindsight_trajectories(self) -> None:
        assert isinstance(self.env, HindsightWrapper)
        self.buffer.extend(self.env.recompute_trajectory(self.trajectory))
        if self.n_goals - 1 and self.trajectory:
            final_states = np.random.randint(1, len(self.trajectory), size=self.n_goals - 1)
            assert isinstance(final_states, np.ndarray)
            for final_state in final_states:
                self.buffer.extend(self.env.recompute_trajectory(self.trajectory,
                                                                 final_state=final_state))

    def reset(self) -> State:
        self.add_hindsight_trajectories()
        return super().reset()

    def vectorize_state(self, state: State) -> np.ndarray:
        assert isinstance(self.env, HindsightWrapper)
        return self.env.vectorize_state(state)


class DoubleBufferHindsightTrainer(DoubleBufferTrainer, HindsightTrainer):
    def __init__(self, env, buffer_size: int, **kwargs):
        assert isinstance(env, HindsightWrapper)
        DoubleBufferTrainer.__init__(self, env=env, buffer_size=buffer_size, **kwargs)

    def reset(self):
        self.add_hindsight_trajectories()
        return DoubleBufferTrainer.reset(self)


class PropagationTrainer(TrajectoryTrainer):
    def add_to_buffer(self, _):
        pass

    def build_agent(self, **kwargs) -> AbstractAgent:
        return super().build_agent(base_agent=PropagationAgent, **kwargs)

    def reset(self) -> State:
        self.buffer.extend(self.step_generator(self.trajectory))
        return super().reset()

    @staticmethod
    def step_generator(trajectory: Iterable[Step]) -> Iterator[PropStep]:
        v2 = 0
        for step in reversed(trajectory):
            v2 = .99 * v2 + step.r
            # noinspection PyProtectedMember
            yield PropStep(v2=v2, **step._asdict())

    def sample_buffer(self):
        return PropStep(*self.buffer.sample(self.batch_size))


class HindsightPropagationTrainer(HindsightTrainer, PropagationTrainer):
    def reset(self) -> State:
        self.add_hindsight_trajectories()
        return PropagationTrainer.reset(self)
