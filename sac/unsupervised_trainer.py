from collections import namedtuple
from collections.__init__ import deque, Counter

import numpy as np
import tensorflow as tf

from environments.hierarchical_wrapper import Hierarchical, HierarchicalAgents
from environments.hindsight_wrapper import HSRHindsightWrapper, Observation
from sac.train import Trainer, HindsightTrainer, Agents, squash_to_space
from sac.utils import Step

BossState = namedtuple('BossState', 'goal action initial_obs initial_value reward')


class UnsupervisedTrainer(Trainer):
    # noinspection PyMissingConstructor
    def __init__(self,
                 env: HSRHindsightWrapper,
                 sess: tf.Session,
                 worker_kwargs: dict,
                 boss_kwargs: dict,
                 boss_freq: int = None,
                 update_worker: bool = True,
                 n_goals: int = None,
                 worker_load_path=None, **kwargs):

        self.update_worker = update_worker
        self.boss_state = None
        self.worker_o1 = None

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
        self.count = Counter(reward=0, episode=0, time_steps=0)
        self.episode_count = None
        self.time_steps = 0

        def boss_preprocess_func(obs, _):
            return Observation(*obs).achieved_goal

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
        var_list = {v.name.replace('worker', 'agent').rstrip(':0'): v
                    for v in worker_vars}
        saver = tf.train.Saver(var_list=var_list)
        if worker_load_path:
            saver.restore(self.sess, worker_load_path)
            print("Model restored from", worker_load_path)
        self.trainers = Hierarchical(boss=boss_trainer, worker=worker_trainer)

        self.agents = Agents(
            act=HierarchicalAgents(
                boss=self.trainers.boss.agents.act,
                worker=self.trainers.worker.agents.act,
                initial_state=0),
            train=HierarchicalAgents(
                boss=self.trainers.boss.agents.train,
                worker=self.trainers.worker.agents.train,
                initial_state=0))

    def boss_turn(self, episodes=None):
        if self.boss_freq:
            if episodes is None:
                episodes = self.episode_count['episode']
            return episodes % self.boss_freq == 0
        return self.time_steps == 0

    def get_actions(self, o1, s, sample: bool):
        assert isinstance(self.env, HSRHindsightWrapper)
        self.worker_o1 = o1.replace(desired_goal=self.env.hsr_env.goal)
        return self.trainers.worker.get_actions(self.worker_o1, s, sample=sample)

    def perform_update(self, _=None):
        if self.update_worker:
            worker_update = {
                f'worker_{k}': v
                for k, v in (self.trainers.worker.perform_update() or {}).items()
            }
        else:
            worker_update = dict()

        self.time_steps += 1
        return {
            **{f'boss_{k}': v
               for k, v in (self.trainers.boss.perform_update(
                train_values=self.agents.act.boss.default_train_values + ['v1']) or {}).items()},
            **worker_update
        }

    def trajectory(self, time_steps: int, final_index=None):
        raise NotImplementedError

    def step(self, action: np.ndarray, render: bool):
        o2, r, t, info = super().step(action=action, render=render)
        self.reward_queue += [r]
        return o2, r, t, info

    def add_to_buffer(self, step: Step):
        self.trainers.worker.buffer.append(step.replace(o1=self.worker_o1))

    def reset(self):
        self.time_steps = 0
        o1 = super().reset()
        self.boss_state = None
        if not self.is_eval_period():
            boss_action = self.trainers.boss.get_actions(o1, 0, sample=False)
            goal_delta = squash_to_space(boss_action.output, space=self.env.goal_space)
            goal = np.clip(o1.achieved_goal + goal_delta,
                           self.env.goal_space.low,
                           self.env.goal_space.high)
            self.env.hsr_env.set_goal(goal)
            if self.boss_state is not None:
                self.trainers.boss.buffer.append(
                    Step(s=0, o1=self.boss_state.initial_obs,
                         a=self.boss_state.action,
                         r=self.boss_state.reward,
                         o2=o1,
                         t=True))  # TODO: what about False?

            v1 = self.agents.act.worker.get_v1(o1=self.trainers.worker.preprocess_func(o1))
            normalized_v1 = v1 / self.agents.act.worker.reward_scale
            self.boss_state = BossState(goal=goal,
                                        action=goal_delta,
                                        initial_obs=o1,
                                        initial_value=normalized_v1,
                                        reward=None)
        return o1

    def run_episode(self, o1, eval_period, render):
        episode_count = super().run_episode(o1=o1,
                                            eval_period=eval_period,
                                            render=render)
        if eval_period:
            import ipdb; ipdb.set_trace()
            return episode_count
        reward = episode_count['reward']
        if self.use_entropy_reward:
            p = self.boss_state.initial_value / .99 ** self.time_steps
            squashed_p = sigmoid(p)
            if reward == 0:
                squashed_p = 1 - squashed_p
            boss_reward = -np.log(squashed_p)
        else:
            rewards = np.array(self.reward_queue)
            boss_reward = np.matmul(self.reward_operator, rewards)[1]

        episode_count['boss_reward'] = boss_reward
        episode_count['initial_value'] = self.boss_state.initial_value
        self.boss_state = self.boss_state._replace(reward=boss_reward)
        print('\nBoss Reward:', boss_reward, '\t Initial Value:', self.boss_state.initial_value)
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
