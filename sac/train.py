import itertools
import time
from collections import Counter, deque, namedtuple
from typing import Optional, Tuple

import gym
import numpy as np
import tensorflow as tf
from gym import Wrapper, spaces
from gym.spaces import Box

from environments.hierarchical_wrapper import (FrozenLakeHierarchicalWrapper,
                                               Hierarchical,
                                               HierarchicalAgents,
                                               HierarchicalWrapper)
from environments.hindsight_wrapper import HindsightWrapper, Observation
from environments.multi_task import MultiTaskEnv
from sac.agent import AbstractAgent, NetworkOutput
from sac.policies import CategoricalPolicy, GaussianPolicy
from sac.replay_buffer import ReplayBuffer
from sac.utils import Obs, Step, create_sess, unwrap_env, vectorize

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
                sess=sess,
                batch_size=None,
                seq_len=1,
                reuse=False,
                action_space=action_space,
                observation_space=observation_space,
                **kwargs),
            train=self.build_agent(
                sess=sess,
                batch_size=batch_size,
                seq_len=seq_len,
                reuse=True,
                action_space=action_space,
                observation_space=observation_space,
                **kwargs))
        self.seq_len = self.agents.act.seq_len
        self.count = Counter(reward=0, episode=0, time_steps=0)
        self.episode_count = Counter()

        obs = env.reset()
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
                        summary.value.add(tag=k, simple_value=self.episode_count[k])
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
        self.episode_count = Counter()
        episode_mean = Counter()
        tick = time.time()
        s = self.agents.act.initial_state
        if render:
            self.env.render()
        for time_steps in itertools.count(1):
            a, s = self.get_actions(o1, s)
            o2, r, t, info = self.step(a)
            if render:
                self.env.render()
            if 'print' in info:
                print('Time step:', time_steps, info['print'])
            if 'log count' in info:
                self.episode_count.update(Counter(info['log count']))
            if 'log mean' in info:
                episode_mean.update(Counter(info['log mean']))
            self.episode_count.update(Counter(reward=r, time_steps=1))
            self.add_to_buffer(Step(s=s, o1=o1, a=a, r=r, o2=o2, t=t))

            if perform_updates:
                episode_mean.update(Counter(self.perform_update()))
            o1 = o2
            episode_mean.update(Counter(fps=1 / float(time.time() - tick)))
            tick = time.time()
            if t:
                for k in episode_mean:
                    self.episode_count[k] = episode_mean[k] / float(time_steps)
                return self.episode_count

    def perform_update(self):
        if self.buffer_full():
            for i in range(self.num_train_steps):
                step = self.agents.act.train_step(self.sample_buffer())
            return {k.replace(' ', '_'): v for k, v in step.items() if np.isscalar(v)}

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

    def step(self, action: np.ndarray) -> Tuple[Obs, float, bool, dict]:
        """ Preprocess action before feeding to env """
        if type(self.action_space) is spaces.Discrete:
            # noinspection PyTypeChecker
            return self.env.step(np.argmax(action))
        else:
            action = np.tanh(action)
            hi, lo = self.action_space.high, self.action_space.low
            # noinspection PyTypeChecker
            return self.env.step((action + 1) / 2 * (hi - lo) + lo)

    def preprocess_obs(self, obs, shape: tuple = None):
        if self.preprocess_func is not None:
            obs = self.preprocess_func(obs, shape)
        return obs

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


class HindsightTrainer(Trainer):
    def __init__(self, env: Wrapper, n_goals: int, **kwargs):
        self.n_goals = n_goals
        self.hindsight_env = unwrap_env(env, lambda e: isinstance(e, HindsightWrapper))
        assert isinstance(self.hindsight_env, HindsightWrapper)
        super().__init__(env=env, **kwargs)

    def add_hindsight_trajectories(self) -> None:
        assert isinstance(self.hindsight_env, HindsightWrapper)
        if self.time_steps() > 0:
            new_trajectory = self.hindsight_env.recompute_trajectory(self.trajectory())
            self.buffer.append(new_trajectory)
        if self.n_goals - 1 and self.time_steps() > 1:
            final_indexes = np.random.randint(1, self.time_steps(), size=self.n_goals - 1)
            assert isinstance(final_indexes, np.ndarray)

            for final_index in final_indexes:
                traj = self.trajectory(final_index)
                new_traj = self.hindsight_env.recompute_trajectory(traj)
                self.buffer.append(new_traj)

    def reset(self) -> Obs:
        self.add_hindsight_trajectories()
        return super().reset()


class MultiTaskTrainer(Trainer):
    def __init__(self, evaluation, env, **kwargs):
        self.eval = evaluation
        self.n = 50000
        self.last_n_rewards = deque(maxlen=self.n)
        self.multi_task_env = unwrap_env(env, lambda e: isinstance(e, MultiTaskEnv))
        super().__init__(env=env, **kwargs)

    def run_episode(self, o1, perform_updates, render):
        env = self.env.unwrapped
        assert isinstance(env, MultiTaskEnv)
        if self.is_eval_period():
            for goal_corner in env.goal_corners:
                o1 = self.reset()
                env.goal = goal_corner + env.goal_size / 2
                count = super().run_episode(
                    o1=o1, perform_updates=perform_updates, render=render)
                for k in count:
                    print(f'{k}: {count[k]}')
            print('Evaluation complete.')
            exit()
        else:
            episode_count = super().run_episode(o1, perform_updates, render)
            self.last_n_rewards.append(episode_count['reward'])
            average_reward = sum(self.last_n_rewards) / self.n
            if average_reward > .96:
                print(f'Reward for last {self.n} episodes: {average_reward}')
                exit()
            return episode_count

    def is_eval_period(self):
        return self.eval


class MultiTaskHindsightTrainer(MultiTaskTrainer, HindsightTrainer):
    pass


BossState = namedtuple('BossState', 'goal action o1')
WorkerState = namedtuple('WorkerState', 'o1 o2')


class HierarchicalTrainer(Trainer):
    # noinspection PyMissingConstructor
    def __init__(self, env, boss_act_freq: int, use_boss_oracle: bool,
                 use_worker_oracle: bool, sess: tf.Session, worker_kwargs, boss_kwargs,
                 correct_boss_action, **kwargs):
        assert isinstance(env, HierarchicalWrapper)
        self.correct_boss_action = correct_boss_action
        self.boss_oracle = use_boss_oracle
        self.worker_oracle = use_worker_oracle
        self.boss_act_freq = boss_act_freq

        self.boss_state = None
        self.worker_o1 = None

        self.env = env
        self.sess = sess
        self.count = Counter(reward=0, episode=0, time_steps=0)
        self.episode_count = Counter()
        self.action_space = env.action_space.worker

        def boss_preprocess_obs(obs, shape=None):
            obs = Observation(*obs)
            return vectorize([obs.achieved_goal, obs.desired_goal], shape)

        def worker_preprocess_obs(obs, shape=None):
            obs = Observation(*obs)
            return vectorize([obs.observation, obs.desired_goal], shape)

        self.trainers = Hierarchical(
            boss=Trainer(
                observation_space=env.observation_space.boss,
                action_space=env.action_space.boss,
                preprocess_func=boss_preprocess_obs,
                env=env,
                sess=sess,
                name='boss',
                **boss_kwargs,
                **kwargs,
            ),
            worker=Trainer(
                observation_space=env.observation_space.worker,
                action_space=env.action_space.worker,
                preprocess_func=worker_preprocess_obs,
                env=env,
                sess=sess,
                name='worker',
                **worker_kwargs,
                **kwargs,
            ))

        self.agents = Agents(
            act=HierarchicalAgents(
                boss=self.trainers.boss.agents.act,
                worker=self.trainers.worker.agents.act,
                initial_state=0),
            train=HierarchicalAgents(
                boss=self.trainers.boss.agents.train,
                worker=self.trainers.worker.agents.train,
                initial_state=0))

    def boss_turn(self):
        return self.time_steps() % self.boss_act_freq == 0

    def get_actions(self, o1, s):
        # boss
        if self.boss_turn():
            if self.boss_oracle:
                action = boss_oracle(self.env)
            else:
                action = self.trainers.boss.get_actions(o1, s).output
            goal = o1.achieved_goal + self.env.boss_action_to_goal_space(action)
            self.boss_state = BossState(goal=goal, action=action, o1=o1)

        # worker
        direction = self.boss_state.goal - o1.achieved_goal
        self.worker_o1 = o1.replace(desired_goal=direction)

        if self.worker_oracle:
            oracle_action = worker_oracle(self.env, direction)
            return NetworkOutput(output=oracle_action, state=0)
        else:
            return self.trainers.worker.get_actions(self.worker_o1, s)

    def perform_update(self):
        return {
            **{
                f'boss_{k}': v
                for k, v in (self.trainers.boss.perform_update() or {}).items()
            },
            **{
                f'worker_{k}': v
                for k, v in (self.trainers.worker.perform_update() or {}).items()
            }
        }

    def trajectory(self, final_index=None):
        raise NotImplemented

    def add_to_buffer(self, step: Step):
        # boss
        if not self.boss_oracle and (self.boss_turn() or step.t):
            boss_action = self.boss_state.action
            if self.correct_boss_action:
                rel_step = step.o2.achieved_goal - self.boss_state.o1.achieved_goal
                boss_action = self.env.goal_to_boss_action_space(rel_step)
            self.trainers.boss.buffer.append(step.replace(
                o1=self.boss_state.o1,
                a=boss_action))

        # worker
        if not self.worker_oracle:
            direction = self.worker_o1.desired_goal
            if not step.t:
                movement = vectorize(step.o2.achieved_goal) - vectorize(step.o1.achieved_goal)
                step = step.replace(r=np.dot(direction, movement))

            step = step.replace(
                o1=self.worker_o1,
                o2=step.o2.replace(desired_goal=direction))
            self.trainers.worker.buffer.append(step)
            self.episode_count.update(Counter(worker_reward=step.r))


def boss_oracle(env: HierarchicalWrapper):
    def alignment(a):
        goal_dir = env._boss_action_to_goal_space(a)
        norm = np.linalg.norm(goal_dir)
        if not np.allclose(norm, 0):
            goal_dir /= norm
        return np.dot(goal_dir, direction)

    direction = env._desired_goal() - env._achieved_goal()
    if isinstance(env.action_space.boss, Box):
        return direction / max(np.linalg.norm(direction), 1e-6)
    actions = list(range(env.action_space.boss.n))
    # alignments = list(map(alignment, actions))
    return env._boss_action_to_goal_space(max(actions, key=alignment))


DIRECTIONS = np.array([
    [0, 0],  # stay
    [0, -1],  # left
    [1, 0],  # down
    [0, 1],  # right
    [-1, 0],  # up
])


def worker_oracle(env: FrozenLakeHierarchicalWrapper, relative_goal: np.ndarray):
    fl = env.frozen_lake_env
    s = fl.from_s(fl.s)
    goal = s + relative_goal

    def in_bounds(new_s):
        return np.all(np.zeros(2) <= new_s) and np.all(
            new_s < np.array([fl.nrow, fl.ncol]))

    def distance_from_goal(i):
        new_s = s + DIRECTIONS[i]
        if in_bounds(new_s) and fl.desc[tuple(new_s)] == b'H':
            return np.inf
        return np.linalg.norm(new_s - goal)

    actions = list(range(env.action_space.worker.n))
    distances = list(map(distance_from_goal, actions))
    action = np.zeros(env.action_space.worker.n)
    best_distances = [i for i in actions if distances[i] == min(distances)]
    i = np.random.choice(best_distances)
    action[i] = 1
    return action
