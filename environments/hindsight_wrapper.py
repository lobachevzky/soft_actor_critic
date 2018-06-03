from abc import abstractmethod
from collections import namedtuple

import gym
import numpy as np
from gym.spaces import Box

from environments.base import distance_between
from environments.pick_and_place import Goal, PickAndPlaceEnv
from sac.utils import Step

State = namedtuple('State', 'observation achieved_goal desired_goal')


def goals_equal(goal1, goal2):
    return all([np.allclose(a, b) for a, b in [(goal1.block, goal2.block),
                                               (goal1.gripper, goal2.gripper)]])


class HindsightWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        vector_state = self.vectorize_state(self.reset())
        self.observation_space = Box(-1, 1, vector_state.shape)

    @abstractmethod
    def _achieved_goal(self):
        raise NotImplementedError

    @abstractmethod
    def _is_success(self, achieved_goal, desired_goal):
        raise NotImplementedError

    @abstractmethod
    def _desired_goal(self):
        raise NotImplementedError

    @staticmethod
    def vectorize_state(state):
        return np.concatenate(state)

    def step(self, action):
        s2, r, t, info = self.env.step(action)
        new_s2 = State(observation=s2,
                       desired_goal=self._desired_goal(),
                       achieved_goal=self._achieved_goal())
        is_success = self._is_success(new_s2.achieved_goal,
                                      new_s2.desired_goal)
        new_t = is_success or t
        new_r = float(is_success)
        info['base_reward'] = r
        return new_s2, new_r, new_t, info

    def reset(self):
        return State(observation=self.env.reset(),
                     desired_goal=self._desired_goal(),
                     achieved_goal=self._achieved_goal())

    def recompute_trajectory(self, trajectory, final_state=-1):
        if not trajectory:
            return ()
        achieved_goal = trajectory[final_state].s2.achieved_goal
        for step in trajectory[:final_state]:
            new_t = self._is_success(step.s2.achieved_goal, achieved_goal)
            r = float(new_t)
            yield Step(
                s1=step.s1._replace(desired_goal=achieved_goal),
                a=step.a,
                r=r,
                s2=step.s2._replace(desired_goal=achieved_goal),
                t=new_t)
            if new_t:
                break


class MountaincarHindsightWrapper(HindsightWrapper):
    """
    new obs is [pos, vel, goal_pos]
    """

    def _achieved_goal(self):
        return self.env.unwrapped.state[0]

    def _is_success(self):
        return self.env.unwrapped.state[0] >= self._desired_goal()

    def _desired_goal(self):
        return 0.45


class PickAndPlaceHindsightWrapper(HindsightWrapper):
    def __init__(self, env):
        super().__init__(env)

    def _is_success(self, achieved_goal, desired_goal):
        return self.env.unwrapped.is_success(achieved_goal, desired_goal)

    def _achieved_goal(self):
        return self.env.unwrapped.achieved_goal()

    def _desired_goal(self):
        return self.env.unwrapped.goal()

    @staticmethod
    def vectorize_state(state):
        return np.concatenate([state.observation, np.concatenate(state.desired_goal)])
