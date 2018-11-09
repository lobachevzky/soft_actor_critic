# stdlib
from collections import Counter, namedtuple

# third party
import numpy as np

# first party
from environments.hindsight_wrapper import Observation
from environments.hsr import distance_between
from sac.replay_buffer import ReplayBuffer
from sac.train import Trainer
from sac.utils import Step

BossState = namedtuple('BossState', 'history delta_tde initial_achieved initial_value '
                                    'reward td_error ')
Key = namedtuple('BufferKey', 'achieved_goal desired_goal')


class UnsupervisedTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.test_sample = None
        self.boss_state = BossState(
            td_error=None,
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
            for i in range(self.n_train_steps):
                # get train and test samples
                sample = self.sample_buffer(batch_size=1)
                self.test_sample.append(sample)
                train_sample = self.sample_buffer()
                test_sample = Step(*self.test_sample.buffer)

                # get pre- and post-td-error
                agent = self.agents.act
                pre_td_error = agent.td_error(step=test_sample)
                train_result = agent.train_step(step=train_sample)
                post_td_error = agent.td_error(step=test_sample)
                delta_tde = np.mean(pre_td_error - post_td_error)
                old_delta_tde = self.boss_state.delta_tde or delta_tde

                if self.boss_state.history is not None:
                    fetch = dict(
                        estimated_delta_tde=agent.estimated_delta_tde,
                        model_loss=agent.model_loss,
                        model_grad=agent.model_grad,
                        train_model=agent.train_model)
                    train_result.update(
                        self.sess.run(
                            fetch,
                            feed_dict={
                                agent.O1: train_sample.o1,
                                agent.A: train_sample.a,
                                agent.R: train_sample.r,
                                agent.O2: train_sample.o2,
                                agent.T: train_sample.t,
                                agent.history: self.boss_state.history,
                                agent.old_delta_tde: np.mean(self.boss_state.delta_tde),
                                agent.delta_tde: delta_tde,
                            }))
                    estimated_delta_tde = train_result['estimated_delta_tde']

                    diff = delta_tde - estimated_delta_tde
                    mean_sq_diff = np.mean(.5 * np.square(diff))
                    # noinspection PyTypeChecker
                    counter.update(
                        td_error=np.mean(post_td_error),
                        estimated_delta_tde=np.mean(estimated_delta_tde),
                        delta_tde=np.mean(delta_tde),
                        diff=np.mean(diff),
                        mean_sq_diff=mean_sq_diff,
                        episodic_delta_tde=np.mean(post_td_error -
                                                   self.boss_state.td_error),
                    )
                history = train_sample.replace(
                    r=train_sample.r.reshape(-1, 1), t=train_sample.t.reshape(-1, 1))

                td_error = train_result['q_error'].reshape(-1, 1)
                history = np.hstack(list(history[1:]) + [td_error])
                self.boss_state = self.boss_state._replace(
                    td_error=post_td_error, history=history, delta_tde=delta_tde)

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

            self.boss_state = self.boss_state._replace(
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
