# stdlib
from collections import Counter, namedtuple

# third party
import numpy as np
import tensorflow as tf

# first party
from environments.hindsight_wrapper import Observation
from environments.hsr import distance_between
from sac.replay_buffer import ReplayBuffer
from sac.train import Trainer
from sac.utils import Step


class BossState(
    namedtuple('BossState', 'history delta_tde initial_achieved initial_value '
                            'reward td_errors ')):
    def replace(self, **kwargs):
        # noinspection PyProtectedMember
        return super()._replace(**kwargs)


Key = namedtuple('BufferKey', 'achieved_goal desired_goal')

Samples = namedtuple('Samples', 'train valid test')


class UnsupervisedTrainer(Trainer):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(**kwargs)
        assert alpha is not None
        self.alpha = alpha
        self.test_sample = None
        self.boss_state = BossState(
            td_errors=None,
            history=None,
            delta_tde=None,
            initial_achieved=None,
            initial_value=None,
            reward=None)

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
                    return Samples(*map(agent.td_error,
                                        Samples(train=train_sample,
                                                valid=valid_sample,
                                                test=Step(
                                                    *self.test_sample.buffer.values))))

                pre = td_errors()
                train_result = agent.train_step(step=train_sample)
                post = td_errors()

                def get_delta(pre, post):
                    return np.mean(pre - post)

                delta = Samples(*[get_delta(*args) for args in zip(pre, post)])

                # new_delta_tde = delta(pre_td_error, post_td_error)
                # old_delta_tde = self.boss_state.delta_tde or new_delta_tde
                # delta_tde = old_delta_tde + self.alpha * (new_delta_tde - old_delta_tde)

                # print(delta(test_pre_td_error, test_post_td_error))

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
                    episodic = Samples(*[get_delta(*args) for args in
                                         zip(self.boss_state.td_errors, post)])

                    counter.update(
                        episodic_train_td_error_delta=episodic.train,
                        episodic_test_td_error_delta=episodic.test,
                        episodic_valid_td_error_delta=episodic.valid,
                    )
                #     # test_td_error=np.mean(test_post_td_error),
                #     # train_td_error=np.mean(train_post_td_error),
                #     estimated_delta_tde=np.mean(estimated),
                #     # delta_tde=delta_tde,
                #     # train_delta_tde=np.mean(train_pre_td_error -
                # train_post_td_error),
                #     # diff=np.mean(delta_tde - estimated_delta_tde),
                #     # episodic_delta_tde=np.mean(test_post_td_error -
                #     #                            self.boss_state.td_error),
                # )
                # history = np.stack([train_result['Q_error'], train_delta_tde], axis=1)
                # td_error = train_result['TDError'].reshape(-1, 1)
                # history = np.hstack(list(history[1:]) + [td_error])
                self.boss_state = self.boss_state.replace(
                    td_errors=post,
                    # history=history,
                    # delta_tde=delta_tde
                )

                for k, v in train_result.items():
                    if np.isscalar(v):
                        counter.update(**{k: v})

        return counter

    def trajectory(self, time_steps: int, final_index=None):
        raise NotImplementedError

    def reset(self):
        o1 = super().reset()
        if not self.is_eval_period():
            # goal_delta = self.trainers.boss.get_actions(o1=o1, s=0, sample=False).output
            # goal = o1.achieved_goal + goal_delta
            goal = self.env.goal_space.sample()
            self.env.hsr_env.set_goal(goal)

            o1 = o1.replace(desired_goal=goal)

            self.boss_state = self.boss_state.replace(
                initial_achieved=Observation(*o1).achieved_goal)
        return o1

    def run_episode(self, o1, eval_period, render):
        episode_count = super().run_episode(o1=o1, eval_period=eval_period, render=render)
        if eval_period:
            return episode_count
        assert np.allclose(o1.desired_goal, self.env.hsr_env.goal)

        goal_distance = distance_between(
            self.boss_state.initial_achieved,
            o1.desired_goal,
        )

        episode_count['goal_distance'] = goal_distance

        print('\nBoss Reward:', self.boss_state.reward, '\t Goal Distance:',
              goal_distance)
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
