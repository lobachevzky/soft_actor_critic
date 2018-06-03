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
    def _achieved_goal(self, obs):
        raise NotImplementedError

    @abstractmethod
    def _achieved_goal2(self):
        raise NotImplementedError

    @abstractmethod
    def _is_success(self, obs, goal):
        raise NotImplementedError

    @abstractmethod
    def _is_success2(self, achieved_goal, desired_goal):
        raise NotImplementedError

    @abstractmethod
    def _desired_goal(self):
        raise NotImplementedError

    @staticmethod
    def vectorize_state(state):
        return np.concatenate(state)

    def _reward(self, state, goal):
        return float(self._is_success(state, goal))

    def step(self, action):
        s2, r, t, info = self.env.step(action)
        achieved_goal2 = self._achieved_goal2()
        achieved_goal = self._achieved_goal(s2)
        assert goals_equal(achieved_goal, achieved_goal2)
        desired_goal = self._desired_goal()
        new_s2 = State(observation=s2,
                       desired_goal=desired_goal,
                       achieved_goal=achieved_goal)
        new_r = self._reward(s2, desired_goal)
        is_success2 = self._is_success2(achieved_goal2, desired_goal)
        assert np.isclose(new_r, float(is_success2))
        new_t = self._is_success(s2, desired_goal) or t
        assert new_t == (is_success2 or t)
        info['base_reward'] = r
        return new_s2, new_r, new_t, info

    def reset(self):
        return State(observation=self.env.reset(),
                     desired_goal=self._desired_goal(),
                     achieved_goal=self._achieved_goal2())

    def recompute_trajectory(self, trajectory, final_state=-1):
        if not trajectory:
            return ()
        achieved_goal = self._achieved_goal(trajectory[final_state].s2.observation)
        assert goals_equal(achieved_goal, trajectory[final_state].s2.achieved_goal)
        for step in trajectory[:final_state]:
            new_t = self._is_success(step.s2.observation, achieved_goal) or step.t
            assert new_t == self._is_success2(step.s2.achieved_goal, achieved_goal)
            r = self._reward(step.s2.observation, achieved_goal)
            assert np.isclose(r, float(new_t))
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

    # TODO: this is cheating
    def step(self, action):
        s, r, t, i = super().step(action)
        return s, max(r, 0), t, i

    def _achieved_goal(self, obs):
        return np.array([obs[0]])

    def _is_success(self, obs, goal):
        return obs[0] >= goal[0]

    def _desired_goal(self):
        return np.array([0.45])


class PickAndPlaceHindsightWrapper(HindsightWrapper):
    def _is_success2(self, achieved_goal, desired_goal):
        geofence = self.env.unwrapped._geofence
        return distance_between(achieved_goal.block, desired_goal.block) < geofence and \
               distance_between(achieved_goal.gripper, desired_goal.gripper) < geofence

    def _achieved_goal2(self):
        return Goal(
            gripper=self.env.unwrapped.gripper_pos(),
            block=self.env.unwrapped.block_pos())

    def __init__(self, env):
        super().__init__(env)

    def _achieved_goal(self, obs):
        return Goal(
            gripper=self.env.unwrapped.gripper_pos(obs),
            block=self.env.unwrapped.block_pos(obs))

    def _is_success(self, obs, goal):
        return self.env.unwrapped.compute_terminal(goal, obs)

    def _desired_goal(self):
        return self.env.unwrapped.goal()

    @staticmethod
    def vectorize_state(state):
        return np.concatenate([state.observation, np.concatenate(state.desired_goal)])
