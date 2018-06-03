from abc import abstractmethod
from collections import namedtuple, Iterable

import gym
import numpy as np
from gym.spaces import Box

from environments.pick_and_place import Goal, PickAndPlaceEnv
from sac.utils import Step

State = namedtuple('State', 'obs goal')


class HindsightWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        vector_state = self.vectorize_state(self.reset())
        self.observation_space = Box(-1, 1, vector_state.shape)

    @abstractmethod
    def _achieved_goal(self, obs):
        raise NotImplementedError

    @abstractmethod
    def _is_success(self, obs, goal):
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
        new_s2 = State(obs=s2, goal=self._desired_goal())
        new_r = self._reward(s2, self._desired_goal())
        new_t = self._is_success(s2, self._desired_goal()) or t
        info['base_reward'] = r
        return new_s2, new_r, new_t, info

    def reset(self):
        return State(obs=self.env.reset(), goal=self._desired_goal())

    def recompute_trajectory(self, trajectory: Iterable, final_step: Step):
        achieved_goal = None
        for step in trajectory:
            if achieved_goal is None:
                achieved_goal = self._achieved_goal(final_step.s2.obs)
            new_t = self._is_success(step.s2.obs, achieved_goal) or step.t
            r = self._reward(step.s2.obs, achieved_goal)
            yield Step(
                s1=State(obs=step.s1.obs, goal=achieved_goal),
                a=step.a,
                r=r,
                s2=State(obs=step.s2.obs, goal=achieved_goal),
                t=new_t)
            if new_t:
                break


class MountaincarHindsightWrapper(HindsightWrapper):
    """
    new obs is [pos, vel, goal_pos]
    """
    #TODO: this is cheating
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
        return np.concatenate([state.obs, np.concatenate(state.goal)])
