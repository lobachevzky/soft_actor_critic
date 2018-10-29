import random
from collections import deque, namedtuple, defaultdict, Counter

import numpy as np
import tensorflow as tf
from gym.spaces import Box

from environments.hierarchical_wrapper import Hierarchical, HierarchicalAgents
from environments.hindsight_wrapper import HSRHindsightWrapper, Observation
from environments.hsr import distance_between
from sac.replay_buffer import ReplayBuffer
from sac.train import Agents, HindsightTrainer, Trainer
from sac.utils import Step

BossState = namedtuple('BossState', 'goal action initial_obs initial_value reward')
Key = namedtuple('BufferKey', 'achieved_goal desired_goal')


class WorkerTrainer(Trainer):
    def perform_update(self):
        return super().perform_update()

    def train_step(self, add_fetch: list = None):
        if add_fetch is None:
            add_fetch = []
        add_fetch.append('delta_td_error')
        return super().train_step(add_fetch=add_fetch)


class DictBuffer(ReplayBuffer):
    def __init__(self):
        super().__init__(maxlen=0)
        self.buffer = defaultdict(0)

    def __getitem__(self, key: Key):
        assert self.buffer is not None
        return self.buffer[key]

    def __setitem__(self, key: Key, value):
        self.buffer[key] = value

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int, seq_len: int = None):
        return random.choices(self.buffer.values(), batch_size)

    def append(self, step: Step):
        o1 = Observation(*step.o1)
        key = Key(tuple(o1.achieved_goal), tuple(o1.desired_goal))
        self[key] = step


# TODO: can we get rid of this?
def boss_preprocess_func(obs, _=None):
    return Observation(*obs).achieved_goal


class UnsupervisedTrainer(Trainer):
    # noinspection PyMissingConstructor
    def __init__(self,
                 env: HSRHindsightWrapper,
                 sess: tf.Session,
                 worker_kwargs: dict,
                 boss_kwargs: dict,
                 boss_freq: int = None,
                 n_train_steps: int = None,
                 update_worker: bool = True,
                 n_goals: int = None,
                 worker_load_path=None,
                 **kwargs):

        self.n_train_steps = n_train_steps
        self.update_worker = update_worker
        self.boss_state = None
        self.worker_o1 = None
        self.boss_o1 = None

        self.boss_freq = boss_freq
        self.reward_queue = deque(maxlen=boss_freq)
        self.use_entropy_reward = boss_freq is None
        if boss_freq:
            X = np.ones((boss_freq, 2))
            X[:, 1] = np.arange(boss_freq)
            self.reward_operator = np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T)

        self.env = env
        self.action_space = env.action_space
        self.sess = sess
        self.episode_count = None

        boss_trainer = Trainer(
            observation_space=env.observation_space,
            action_space=env.goal_space,
            preprocess_func=boss_preprocess_func,
            env=env,
            sess=sess,
            name='boss',
            **boss_kwargs,
            **kwargs,
        )

        worker_kwargs = dict(
            observation_space=env.observation_space,
            action_space=env.action_space,
            env=env,
            sess=sess,
            name='worker',
            **worker_kwargs,
            **kwargs)
        if n_goals is None:
            worker_trainer = Trainer(**worker_kwargs)
        else:
            worker_trainer = HindsightTrainer(n_goals=n_goals, **worker_kwargs)

        worker_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='worker')
        var_list = {
            v.name.replace('worker', 'agent').rstrip(':0'): v
            for v in worker_vars
        }
        saver = tf.train.Saver(var_list=var_list)
        if worker_load_path:
            saver.restore(self.sess, worker_load_path)
            print("Model restored from", worker_load_path)

        self.trainers = Hierarchical(boss=boss_trainer, worker=worker_trainer)

        self.global_step = self.trainers.worker.global_step
        self.episode_time_step = self.trainers.worker.episode_time_step
        self.increment_global_step = self.trainers.worker.increment_global_step

        self.agents = Agents(
            act=HierarchicalAgents(
                boss=self.trainers.boss.agents.act,
                worker=self.trainers.worker.agents.act,
                initial_state=0),
            train=HierarchicalAgents(
                boss=self.trainers.boss.agents.train,
                worker=self.trainers.worker.agents.train,
                initial_state=0))

    def get_actions(self, o1, s, sample: bool):
        assert isinstance(self.env, HSRHindsightWrapper)
        self.worker_o1 = o1.replace(desired_goal=self.env.hsr_env.goal)
        return self.trainers.worker.get_actions(o1=self.worker_o1, s=s, sample=sample)

    def sample_buffer(self) -> Step:
        worker = self.trainers.worker
        return Step(*worker.buffer.sample(worker.batch_size, seq_len=worker.seq_len))

    def perform_update(self):
        counter = Counter()
        worker_trainer = self.trainers.worker
        worker_agent = self.agents.act.worker

        def preprocess_obs(obs):
            return worker_trainer.preprocess_obs(
                obs, shape=[worker_trainer.batch_size, -1])

        def preprocess_sample(sample: Step):
            return sample.replace(
                o1=preprocess_obs(sample.o1),
                o2=preprocess_obs(sample.o2)
            )

        batch_size = worker_trainer.batch_size
        if len(worker_trainer.buffer) >= batch_size:
            for i in range(self.n_train_steps):
                train_sample = self.sample_buffer()
                test_sample = self.sample_buffer()

                o1 = Observation(*train_sample.o1)
                goal = o1.desired_goal

                train_sample = preprocess_sample(train_sample)
                test_sample = preprocess_sample(test_sample)

                pre_td_error = worker_agent.td_error(test_sample)

                worker_step = worker_agent.train_step(train_sample)
                post_td_error = worker_agent.td_error(test_sample)
                self.boss_state = self.boss_state._replace(reward=post_td_error -
                                                                pre_td_error)

                ones = np.ones(batch_size)
                boss_step = self.agents.act.boss.train_step(
                    Step(s=0,
                         o1=train_sample.s,
                         a=goal - train_sample.s,
                         r=ones * self.boss_state.reward,
                         o2=np.zeros_like(train_sample.s),
                         # dummy
                         t=ones,
                         # True
                         ))

                def count(step: Step, prefix: str) -> dict:
                    return {prefix + k.replace(' ', '_'): v
                            for k, v in step.items() if np.isscalar(v)}

                counter.update(
                    Counter(**count(worker_step, 'worker_'),
                            **count(boss_step, 'boss_'), ))
        return counter

    def trajectory(self, time_steps: int, final_index=None):
        raise NotImplementedError

    def step(self, action: np.ndarray, render: bool):
        o2, r, t, info = super().step(action=action, render=render)
        self.reward_queue += [r]
        return o2, r, t, info

    def add_to_buffer(self, step: Step):
        self.trainers.worker.buffer.append(step.replace(
            s=boss_preprocess_func(self.boss_state.initial_obs),
            o1=self.worker_o1))

    def reset(self):
        o1 = super().reset()
        if not self.is_eval_period():
            goal_delta = self.trainers.boss.get_actions(o1, 0, sample=False).output
            # goal = o1.achieved_goal + goal_delta
            goal = self.env.goal_space.sample()
            self.env.hsr_env.set_goal(goal)

            v1 = self.agents.act.worker.get_v1(
                o1=self.trainers.worker.preprocess_func(o1))
            normalized_v1 = v1 / self.agents.act.worker.reward_scale
            self.boss_state = BossState(
                goal=goal,
                action=goal_delta,
                initial_obs=o1,
                initial_value=normalized_v1,
                reward=None)
        return o1

    def run_episode(self, o1, eval_period, render):
        episode_count = super().run_episode(o1=o1, eval_period=eval_period, render=render)
        if eval_period:
            return episode_count

        episode_count['initial_value'] = self.boss_state.initial_value
        episode_count['goal_distance'] = distance_between(self.worker_o1.achieved_goal,
                                                          self.env.hsr_env.goal)

        print('\nBoss Reward:', self.boss_state.reward, '\t Initial Value:',
              self.boss_state.initial_value)
        return episode_count


def sigmoid(x):
    return (np.tanh(x * 2 - 1) + 1) / 2


def regression_slope1(Y):
    Y = np.array(Y)
    X = np.ones((Y.size, 2))
    X[:, 1] = np.arange(Y.size)
    return np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))


def regression_slope2(Y):
    Y = np.array(Y)
    X = np.arange(Y.size)
    normalized_X = X - X.mean()
    return np.sum(normalized_X * Y) / np.sum(normalized_X ** 2)
