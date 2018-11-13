# stdlib
from collections import Counter, namedtuple, deque

# third party
import numpy as np
import tensorflow as tf

# first party
from gym.spaces import Box

from environments.hindsight_wrapper import Observation
from environments.hsr import distance_between, HSREnv
from sac.replay_buffer import ReplayBuffer
from sac.train import Trainer
from sac.utils import Step, unwrap_env


class BossState(
        namedtuple(
            'BossState', 'history delta_tde initial_achieved initial_value '
            'reward td_errors cumulative_delta_td_error')):
    def replace(self, **kwargs):
        # noinspection PyProtectedMember
        return super()._replace(**kwargs)


Samples = namedtuple('Samples', 'train valid test')


class UnsupervisedTrainer(Trainer):
    def __init__(self, episodes_per_goal: int, **kwargs):
        super().__init__(**kwargs)
        self.episodes_per_goal = episodes_per_goal
        self.hsr_env = unwrap_env(self.env, lambda e: isinstance(e, HSREnv))
        self.reward_history = []
        self.test_sample = None
        if episodes_per_goal == 1:
            self.lin_regress_op = None
        else:
            x = np.stack([np.ones(episodes_per_goal),
                          np.arange(episodes_per_goal)],
                         axis=1)
            self.lin_regress_op = np.linalg.inv(x.T @ x) @ x.T
        self.boss_state = BossState(
            td_errors=None,
            history=None,
            delta_tde=None,
            initial_achieved=None,
            initial_value=None,
            reward=None,
            cumulative_delta_td_error=0,
        )

        self.double_goal_space = Box(
            low=self.hsr_env.goal_space.low * 1.1,
            high=self.hsr_env.goal_space.high * 1.1,
        )

    def perform_update(self):
        counter = Counter()

        if self.buffer_ready():
            if self.test_sample is None:
                self.test_sample = ReplayBuffer(maxlen=self.batch_size)
                self.test_sample.extend(self.sample_buffer())
            agent = self.agents.act

            for i in range(self.n_train_steps):
                # get samples
                train_sample = self.sample_buffer()
                valid_sample = self.sample_buffer()
                self.test_sample.append(self.sample_buffer(batch_size=1))

                def td_errors():
                    return Samples(*map(
                        agent.td_error,
                        Samples(
                            train=train_sample,
                            valid=valid_sample,
                            test=Step(*self.test_sample.buffer.values))))

                pre = td_errors()
                train_result = agent.train_step(step=train_sample)
                post = td_errors()

                def get_delta(pre, post):
                    return np.mean(pre - post)

                delta = Samples(*[get_delta(*args) for args in zip(pre, post)])

                # noinspection PyTypeChecker
                counter.update(
                    pre_train_td_error=np.mean(pre.train),
                    pre_test_td_error=np.mean(pre.test),
                    pre_valid_td_error=np.mean(pre.valid),
                    post_train_td_error=np.mean(post.train),
                    post_test_td_error=np.mean(post.test),
                    post_valid_td_error=np.mean(post.valid),
                    train_td_error_delta=delta.train,
                    test_td_error_delta=delta.test,
                    valid_td_error_delta=delta.valid,
                )

                if self.boss_state.td_errors is not None:
                    episodic = Samples(*[
                        get_delta(*args) for args in zip(self.boss_state.td_errors, post)
                    ])

                    counter.update(
                        episodic_train_td_error_delta=episodic.train,
                        episodic_test_td_error_delta=episodic.test,
                        episodic_valid_td_error_delta=episodic.valid,
                    )
                cumulative = self.boss_state.cumulative_delta_td_error
                self.boss_state = self.boss_state.replace(
                    td_errors=post, cumulative_delta_td_error=cumulative + post.train)

                for k, v in train_result.items():
                    if np.isscalar(v):
                        counter.update(**{k: v})

        return counter

    def trajectory(self, time_steps: int, final_index=None):
        raise NotImplementedError

    def run_episode(self, o1, eval_period, render):
        episode_count = super().run_episode(o1=o1, eval_period=eval_period, render=render)

        if eval_period:
            self.hsr_env.set_goal(self.hsr_env.goal_space.sample())
            return episode_count

        self.reward_history.append(episode_count['reward'])
        if len(self.reward_history) == self.episodes_per_goal:
            if self.episodes_per_goal > 1:
                _, episode_count['reward_delta'] = self.lin_regress_op @ np.array(
                    self.reward_history)

            in_range = self.hsr_env.goal_space.contains(self.hsr_env.goal)
            positive_delta = episode_count['reward_delta'] > 0
            agreement = (in_range and positive_delta) or not (in_range or positive_delta)
            failure_to_learn = in_range and not positive_delta
            the_impossible = not in_range and positive_delta

            agent = self.agents.act
            train_result = self.sess.run(
                dict(
                    model_loss=agent.model_loss,
                    model_grad=agent.model_grad,
                    op=agent.train_model),
                feed_dict={
                    agent.goal: self.hsr_env.goal,
                    agent.model_target: positive_delta,
                })

            # choose new goal:
            goal = self.double_goal_space.sample()
            self.hsr_env.set_goal(goal)
            self.reward_history = []

            self.boss_state = self.boss_state.replace(cumulative_delta_td_error=0, )
            episode_count.update(
                dict(
                    in_range=in_range,
                    agreement=agreement,
                    failure_to_learn=failure_to_learn,
                    the_impossible=the_impossible))
            for k, v in train_result.items():
                if np.isscalar(v):
                    episode_count.update(**{k: v})

        # goal_distance = distance_between(
        #     self.boss_state.initial_achieved,
        #     o1.goal,
        # )
        #
        # episode_count['goal_distance'] = goal_distance

        print('\nBoss Reward:', self.boss_state.reward)
        # '\t Goal Distance:', goal_distance)
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
    return np.sum(normalized_X * Y) / np.sum(normalized_X**2)
