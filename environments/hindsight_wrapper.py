from abc import abstractmethod
from collections import namedtuple
from typing import Iterable
import numpy as np

import gym
from gym.spaces import Box

from environments.mujoco import distance_between
from environments.pick_and_place import PickAndPlaceEnv
from sac.utils import Step, vectorize, unwrap_env

Goal = namedtuple('Goal', 'gripper block')


class Observation(namedtuple('Obs', 'observation achieved_goal desired_goal')):
    def replace(self, **kwargs):
        return super()._replace(**kwargs)


class HindsightWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        vector_state = self.preprocess_obs(self.reset())
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

    def step(self, action):
        s2, r, t, info = self.env.step(action)
        new_s2 = Observation(
            observation=s2,
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())
        return new_s2, r, t, info

    def reset(self):
        return Observation(
            observation=self.env.reset(),
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())

    def recompute_trajectory(self, trajectory: Iterable, final_step: Step):
        achieved_goal = None
        for step in trajectory:
            if achieved_goal is None:
                achieved_goal = final_step.s2.achieved_goal
            new_t = self._is_success(step.s2.achieved_goal, achieved_goal)
            r = float(new_t)
            yield Step(
                s1=step.s1.replace(desired_goal=achieved_goal),
                a=step.a,
                r=r,
                s2=step.s2.replace(desired_goal=achieved_goal),
                t=new_t)
            if new_t:
                break

    def preprocess_obs(self, obs, shape: tuple = None):
        obs = Observation(*obs)
        obs = [obs.observation, obs.desired_goal]
        return vectorize(obs, shape=shape)


class MountaincarHindsightWrapper(HindsightWrapper):
    """
    new obs is [pos, vel, goal_pos]
    """
    def step(self, action):
        s2, r, t, info = super().step(action)
        return s2, max([0, r]), t, info

    def _achieved_goal(self):
        return self.env.unwrapped.state[0]

    def _is_success(self, achieved_goal, desired_goal):
        return self.env.unwrapped.state[0] >= self._desired_goal()

    def _desired_goal(self):
        return 0.45


class PickAndPlaceHindsightWrapper(HindsightWrapper):
    def __init__(self, env, geofence):
        super().__init__(env)
        self.pap_env = unwrap_env(env, lambda e: isinstance(e, PickAndPlaceEnv))
        self._geofence = geofence
        self.observation_space = Box(
            low=vectorize(
                Observation(
                    observation=env.observation_space.low,
                    desired_goal=Goal(self.goal_space.low, self.goal_space.low),
                    achieved_goal=None)),
            high=vectorize(
                Observation(
                    observation=env.observation_space.high,
                    desired_goal=Goal(self.goal_space.high, self.goal_space.high),
                    achieved_goal=None)))

    @property
    def goal_space(self):
        return Box(low=np.array([-.14, -.2240, .4]), high=np.array([.11, .2241, .921]))

    def _is_success(self, achieved_goal, desired_goal):
        geofence = self.env.unwrapped.geofence
        return distance_between(achieved_goal.block, desired_goal.block) < geofence and \
            distance_between(achieved_goal.gripper,
                             desired_goal.gripper) < geofence

    def _achieved_goal(self):
        return Goal(
            gripper=self.env.unwrapped.gripper_pos(),
            block=self.env.unwrapped.block_pos())

    def _desired_goal(self):
        return self.env.unwrapped.goal()


