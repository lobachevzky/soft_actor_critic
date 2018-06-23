import functools
from abc import abstractmethod
from collections import namedtuple
from typing import Iterable, List

import gym
import numpy as np
from gym.spaces import Box

from environments.mujoco import distance_between
from environments.pick_and_place import Goal
from sac.utils import Step


def get_size(x):
    if np.isscalar(x):
        return 1
    return sum(map(get_size, x))


def assign_to_vector(x, vector: np.ndarray):
    if isinstance(x, np.ndarray) or np.isscalar(x):
        vector[:, :] = x
    else:
        sizes = np.array([get_size(_x) for _x in x])
        sizes = np.cumsum(sizes / vector.shape[0], dtype=int)
        for _x, start, stop in zip(x, [0] + list(sizes), sizes):
            assign_to_vector(_x, vector[:, start: stop])


class State(namedtuple('State', 'observation achieved_goal desired_goal')):
    def replace(self, **kwargs):
        return super()._replace(**kwargs)


class HindsightWrapper(gym.Wrapper):
    def __init__(self, env):
        self.state_vectors = {}
        super().__init__(env)
        s = self.reset()
        self.batch1size = get_size(s)
        vector_state = self.vectorize_state(s)
        self.observation_space = Box(-1, 1, vector_state.shape[1:])

    @abstractmethod
    def _achieved_goal(self):
        raise NotImplementedError

    @abstractmethod
    def _is_success(self, achieved_goal, desired_goal):
        raise NotImplementedError

    @abstractmethod
    def _desired_goal(self):
        raise NotImplementedError

    def vectorize_state(self, state):
        if isinstance(state, np.ndarray):
            return state
        size = get_size(state)
        # if size not in self.state_vectors:
        vector = np.zeros(size).reshape(-1, self.batch1size)
        # self.state_vectors[size] = vector
        # vector = self.state_vectors[size]
        assert isinstance(vector, np.ndarray)
        assign_to_vector(x=state, vector=vector)
        return vector

    def step(self, action):
        o2, r, t, info = self.env.step(action)
        new_o2 = State(
            observation=o2,
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())
        is_success = self._is_success(new_o2.achieved_goal, new_o2.desired_goal)
        new_t = is_success or t
        new_r = float(is_success)
        info['base_reward'] = r
        return new_o2, new_r, new_t, info

    def reset(self):
        return State(
            observation=self.env.reset(),
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())

    def recompute_trajectory(self, trajectory: Iterable, final_step: Step):
        achieved_goal = None
        for step in trajectory:
            if achieved_goal is None:
                achieved_goal = final_step.o2.achieved_goal
            new_t = self._is_success(step.o2.achieved_goal, achieved_goal)
            r = float(new_t)
            yield Step(
                s=step.s,
                o1=step.o1.replace(desired_goal=achieved_goal),
                a=step.a,
                r=r,
                o2=step.o2.replace(desired_goal=achieved_goal),
                t=new_t)
            if new_t:
                break


class MountaincarHindsightWrapper(HindsightWrapper):
    """
    new obs is [pos, vel, goal_pos]
    """

    def _achieved_goal(self):
        return self.env.unwrapped.state[0]

    def _is_success(self, achieved_goal, desired_goal):
        return self.env.unwrapped.state[0] >= self._desired_goal()

    def _desired_goal(self):
        return 0.45


class PickAndPlaceHindsightWrapper(HindsightWrapper):
    def __init__(self, env):
        super().__init__(env)

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


class MultiTaskHindsightWrapper(PickAndPlaceHindsightWrapper):
    def __init__(self, env):
        super().__init__(env)

    def recompute_trajectory(self, reverse_trajectory: Iterable, final_step: Step):
        achieved_goals = []
        last_goal = True
        for step in reverse_trajectory:
            assert isinstance(step, Step)
            achieved_goal = step.o2.achieved_goal

            if last_goal:
                last_goal = False
                if np.random.uniform(0, 1) < .1:
                    achieved_goals.append(achieved_goal)

            block_lifted = achieved_goal.block[2] > self.env.unwrapped.lift_height
            in_box = achieved_goal.block[1] > .1 and not block_lifted
            if block_lifted or in_box:
                achieved_goals.append(achieved_goal)

            for achieved_goal in achieved_goals:
                new_t = self._is_success(
                    achieved_goal=step.o2.achieved_goal, desired_goal=achieved_goal)
                r = float(new_t)
                yield Step(
                    o1=step.o1.replace(desired_goal=achieved_goal),
                    a=step.a,
                    r=r,
                    o2=step.o2.replace(desired_goal=achieved_goal),
                    t=new_t)
