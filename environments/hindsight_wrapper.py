from abc import abstractmethod
from collections import namedtuple
from copy import deepcopy

import gym
import numpy as np
from gym import spaces
from gym.spaces import Box

import environments.hsr
from environments.frozen_lake import FrozenLakeEnv
from environments.hsr import HSREnv, distance_between
from environments.lift import LiftEnv
from environments.shift import ShiftEnv
from sac.array_group import ArrayGroup
from sac.utils import Step, unwrap_env, vectorize

Goal = namedtuple('Goal', 'gripper block')


class Observation(namedtuple('Obs', 'observation achieved_goal desired_goal')):
    def replace(self, **kwargs):
        return super()._replace(**kwargs)


class HindsightWrapper(gym.Wrapper):
    @abstractmethod
    def _achieved_goal(self):
        raise NotImplementedError

    @abstractmethod
    def _is_success(self, achieved_goal, desired_goal):
        raise NotImplementedError

    @abstractmethod
    def _desired_goal(self):
        raise NotImplementedError

    def _add_goals(self, env_obs):
        observation = Observation(
            observation=env_obs,
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())
        assert self.observation_space.contains(observation)
        return observation

    def step(self, action):
        o2, r, t, info = self.env.step(action)
        return self._add_goals(o2), r, t, info

    def reset(self):
        return self._add_goals(super().reset())

    def recompute_trajectory(self, trajectory: Step):
        trajectory = Step(*deepcopy(trajectory))

        # get values
        o1 = Observation(*trajectory.o1)
        o2 = Observation(*trajectory.o2)
        achieved_goal = ArrayGroup(o2.achieved_goal)[-1]

        # perform assignment
        ArrayGroup(o1.desired_goal)[:] = achieved_goal
        ArrayGroup(o2.desired_goal)[:] = achieved_goal
        trajectory.r[:] = self._is_success(o2.achieved_goal, o2.desired_goal)
        trajectory.t[:] = np.logical_or(trajectory.t, trajectory.r)

        first_terminal = np.flatnonzero(trajectory.t)[0]
        return ArrayGroup(trajectory)[:first_terminal + 1]  # include first terminal

    def preprocess_obs(self, obs, shape: tuple = None):
        obs = Observation(*obs)
        obs = [obs.observation, obs.desired_goal]
        return vectorize(obs, shape=shape)


class MountaincarHindsightWrapper(HindsightWrapper):
    """
    new obs is [pos, vel, goal_pos]
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=vectorize([self.observation_space.low, env.unwrapped.min_position]),
            high=vectorize([self.observation_space.high, env.unwrapped.max_position]))

    def step(self, action):
        o2, r, t, info = super().step(action)
        is_success = self._is_success(o2.achieved_goal, o2.desired_goal)
        new_t = is_success or t
        new_r = float(is_success)
        info['base_reward'] = r
        return o2, new_r, new_t, info

    def _achieved_goal(self):
        return self.env.unwrapped.state[0]

    def _desired_goal(self):
        return 0.45

    def _is_success(self, achieved_goal, desired_goal):
        return achieved_goal >= desired_goal


class HSRHindsightWrapper(HindsightWrapper):
    def __init__(self, env, geofence):
        super().__init__(env)
        self.hsr_env = unwrap_env(env, lambda e: isinstance(e, HSREnv))
        self._geofence = geofence
        self.observation_space = spaces.Tuple(
            Observation(
                observation=self.hsr_env.observation_space,
                desired_goal=self.goal_space,
                achieved_goal=self.goal_space,
            ))

    def _is_success(self, achieved_goal, desired_goal):
        achieved_goal = Goal(*achieved_goal).block
        desired_goal = Goal(*desired_goal).block
        return distance_between(achieved_goal, desired_goal) < self._geofence

    def _achieved_goal(self):
        return Goal(
            gripper=self.hsr_env.gripper_pos(), block=self.hsr_env.block_pos())

    def _desired_goal(self):
        assert isinstance(self.hsr_env, LiftEnv)
        goal = self.hsr_env.initial_block_pos.copy()
        goal[2] += self.hsr_env.min_lift_height
        return Goal(gripper=goal, block=goal)

    @property
    def goal_space(self):
        return spaces.Tuple(
            Goal(
                gripper=Box(-np.inf, np.inf, (3,)),
                block=Box(
                    low=np.array([-.14, -.2240, .4]) - 1,
                    high=np.array([.11, .2241, .921]) + 1)))


class FrozenLakeHindsightWrapper(HindsightWrapper):
    def __init__(self, env):
        self.frozen_lake_env = unwrap_env(env, lambda e: isinstance(e, FrozenLakeEnv))
        super().__init__(env)

    def _achieved_goal(self):
        fl_env = self.frozen_lake_env
        return np.array([fl_env.s // fl_env.nrow, fl_env.s % fl_env.ncol])

    def _is_success(self, achieved_goal, desired_goal):
        return (achieved_goal == desired_goal).prod(axis=-1)

    def _desired_goal(self):
        return self.frozen_lake_env.goal_vector()

    def step(self, action):
        o2, r, t, info = self.env.step(action)
        new_o2 = Observation(
            observation=np.array(o2.observation),
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())
        return new_o2, r, t, info

    def reset(self):
        return Observation(
            observation=np.array(self.env.reset().observation),
            desired_goal=self._desired_goal(),
            achieved_goal=self._achieved_goal())
