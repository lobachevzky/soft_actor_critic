from abc import abstractmethod
from collections import namedtuple
from copy import deepcopy
from typing import Optional

import gym
import numpy as np
from gym.spaces import Box

from sac.array_group import ArrayGroup
from sac.utils import Step, vectorize
import itertools

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

    def step(self, action):
        s2, r, t, info = self.env.step(action)
        new_s2 = Observation(observation=s2,
                             desired_goal=self._desired_goal(),
                             achieved_goal=self._achieved_goal())
        is_success = self._is_success(new_s2.achieved_goal,
                                      new_s2.desired_goal)
        new_t = is_success or t
        new_r = float(is_success)
        info['base_reward'] = r
        return new_s2, new_r, new_t, info

    def reset(self):
        return Observation(observation=self.env.reset(),
                           desired_goal=self._desired_goal(),
                           achieved_goal=self._achieved_goal())

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

    def old_recompute_trajectory(self, trajectory, final_state=-1, debug=False):
        trajectory = list(trajectory)
        if not trajectory:
            return ()
        achieved_goal = trajectory[final_state].o2.achieved_goal
        for i in itertools.count():
            try:
                step = trajectory[i]
            except IndexError:
                if debug:
                    import ipdb; ipdb.set_trace()
                break
            new_t = self._is_success(step.o2.achieved_goal, achieved_goal)
            r = float(new_t)
            if step.t:
                assert new_t
            yield Step(
                s=None,
                o1=step.o1._replace(desired_goal=achieved_goal),
                a=step.a,
                r=r,
                o2=step.o2._replace(desired_goal=achieved_goal),
                t=new_t)
            if new_t:
                if debug:
                    import ipdb; ipdb.set_trace()
                break


class MountaincarHindsightWrapper(HindsightWrapper):
    """
    new obs is [pos, vel, goal_pos]
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=vectorize([self.observation_space.low, env.unwrapped.min_position]),
            high=vectorize([self.observation_space.high, env.unwrapped.max_position])
        )

    def _achieved_goal(self):
        return self.env.unwrapped.state[0]

    def _desired_goal(self):
        return 0.45

    def _is_success(self, achieved_goal, desired_goal):
        return achieved_goal >= desired_goal

    @staticmethod
    def old_vectorize_state(state):
        state = Observation(*state)
        return np.append(state.observation, state.desired_goal)

    def preprocess_obs(self, obs, shape: Optional[tuple] = None):
        obs = Observation(*obs)
        obs = [obs.observation, obs.desired_goal]
        return vectorize(obs, shape=shape)


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
    def old_vectorize_state(state):
        return np.concatenate([state.observation, np.concatenate(state.desired_goal)])
