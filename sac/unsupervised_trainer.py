# stdlib
from collections import Counter, namedtuple

# third party
import numpy as np

# first party
from environments.hindsight_wrapper import Observation
from environments.hsr import distance_between
from sac.train import Trainer
from sac.utils import Step

BossState = namedtuple('BossState', 'history delta_tde initial_achieved initial_value '
                                    'reward')
Key = namedtuple('BufferKey', 'achieved_goal desired_goal')


class UnsupervisedTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.boss_state = BossState(
            history=None,
            delta_tde=None,
            initial_achieved=None,
            initial_value=None,
            reward=None
        )

    def perform_update(self):
        counter = Counter()

        if self.buffer_ready():
            for i in range(self.n_train_steps):
                # sample buffer
                test_sample = self.sample_buffer()
                train_sample = self.sample_buffer()

                # get pre- and post-td-error
                agent = self.agents.act
                pre_td_error = agent.td_error(step=test_sample)
                train_values = agent.train_step(step=train_sample)
                post_td_error = agent.td_error(step=test_sample)
                delta_tde = pre_td_error - post_td_error

                if self.boss_state.history is not None:
                    fetch = dict(
                        model_loss=agent.model_loss,
                        model_grad=agent.model_grad,
                        train_model=agent.train_model)
                    train_result = self.sess.run(
                        list(fetch.values()),
                        feed_dict={agent.O1:            test_sample.o1,
                                   agent.A:             test_sample.a,
                                   agent.history:       self.boss_state.history,
                                   agent.old_delta_tde: self.boss_state.delta_tde,
                                   agent.delta_tde:     delta_tde, })
                    for k, v in zip(fetch.keys(), train_result):
                        train_values[k] = v

                for k, v in train_values.items():
                    if np.isscalar(v):
                        counter.update(**{k: v})

            self.boss_state = self.boss_state._replace(
                history=np.hstack([test_sample.o1, test_sample.a]),
                delta_tde=delta_tde
            )
        return counter

    def trajectory(self, time_steps: int, final_index=None):
        raise NotImplementedError

    def reset(self):
        o1 = super().reset()
        if not self.is_eval_period():
            # goal_delta = self.trainers.boss.get_actions(o1=o1, s=0, sample=False).output
            # goal = o1.achieved_goal + goal_delta
            old_goal = self.env.hsr_env.goal.copy()
            goal = self.env.goal_space.sample()
            goal_delta = goal - old_goal
            self.env.hsr_env.set_goal(goal)

            o1 = o1.replace(desired_goal=goal)

            # v1 = self.agents.act.get_v1(
            #     o1=self.preprocess_func(o1))
            # normalized_v1 = v1 / self.agents.act.reward_scale
            self.boss_state = self.boss_state._replace(
                initial_achieved=Observation(*o1).achieved_goal)
                # initial_value=normalized_v1,
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

        # episode_count['initial_value'] = self.boss_state.initial_value
        episode_count['goal_distance'] = goal_distance
        episode_count['boss_reward'] = self.boss_state.reward

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
