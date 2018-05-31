from abc import abstractmethod
from collections import namedtuple

import gym
import numpy as np
from gym.spaces import Box

from environments.base import distance_between
from environments.pick_and_place import Goal
from sac.utils import Step


class State(namedtuple('State', 'achieved_goal desired_goal observation')):
    def replace(self, **kwargs):
        return super()._replace(**kwargs)


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
        new_t = self._is_success(self._achieved_goal(), self._desired_goal())
        new_r = float(new_t)
        return new_s2, new_r, t or new_t, info

    def reset(self):
        observation = self.env.reset()
        return State(achieved_goal=self._achieved_goal(),
                     desired_goal=self._desired_goal(),
                     observation=observation)

    def recompute_trajectory(self, trajectory, final_index=-1):
        if not trajectory:
            return ()
        achieved_goal = trajectory[final_index].s2.achieved_goal
        for step in trajectory[:final_index]:
            new_t = self._is_success(step.s2.achieved_goal, achieved_goal) or step.t
            r = float(new_t)
            s1 = step.s1.replace(desired_goal=achieved_goal)
            s2 = step.s2.replace(desired_goal=achieved_goal)
            yield Step(s1=s1, a=step.a, r=r, s2=s2, t=new_t)
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

    def _achieved_goal(self):
        return self.env.unwrapped.state[0]

    def _desired_goal(self):
        return 0.45

    def _is_success(self, achieved_goal, desired_goal):
        return achieved_goal >= desired_goal

    @staticmethod
    def vectorize_state(state: State):
        return np.append(state.observation, state.desired_goal)


class PickAndPlaceHindsightWrapper(HindsightWrapper):
    def __init__(self, env):
        super().__init__(env)

    def _achieved_goal(self):
        return Goal(
            gripper=self.env.unwrapped.gripper_pos(),
            block=self.env.unwrapped.block_pos())

    def _is_success(self, achieved_goal, desired_goal):
        geofence = self.env.unwrapped.geofence
        gripper_distance = distance_between(achieved_goal.gripper, desired_goal.gripper)
        block_distance = distance_between(achieved_goal.block, desired_goal.block)
        return gripper_distance < geofence and block_distance < geofence

    def _desired_goal(self):
        return self.env.unwrapped.goal()

    @staticmethod
    def vectorize_state(state: State):
        return np.concatenate([state.observation,
                               np.concatenate(state.desired_goal)])

