import gym
import numpy as np
from abc import abstractmethod
from gym.spaces import Box

from environments.pick_and_place import Goal, PickAndPlaceEnv
from sac.utils import Step, State


class HindsightWrapper(gym.Wrapper):
    def __init__(self, env, default_reward=0):
        self._default_reward = default_reward
        super().__init__(env)
        vector_state = self.vectorize_state(self.reset())
        self.observation_space = Box(-1, 1, vector_state.shape)

    @abstractmethod
    def achieved_goal(self, obs):
        raise NotImplementedError

    @abstractmethod
    def at_goal(self, obs, goal):
        raise NotImplementedError

    @abstractmethod
    def desired_goal(self):
        raise NotImplementedError

    def vectorize_state(self, state):
        return np.concatenate(state)

    def _reward(self, state, goal):
        return 1 if self.at_goal(state, goal) else self._default_reward

    def step(self, action):
        s2, r, t, info = self.env.step(action)
        new_s2 = State(obs=s2, goal=self.desired_goal())
        new_r = self._reward(s2, self.desired_goal())
        new_t = self.at_goal(s2, self.desired_goal()) or t
        info['base_reward'] = r
        return new_s2, new_r, new_t, info

    def reset(self):
        return State(obs=self.env.reset(), goal=self.desired_goal())

    def recompute_trajectory(self, trajectory, final_state=-1):
        if not trajectory:
            return ()
        achieved_goal = self.achieved_goal(trajectory[final_state].s2.obs)
        for step in trajectory[:final_state]:
            new_t = self.at_goal(step.s2.obs, achieved_goal) or step.t
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

    def achieved_goal(self, obs):
        return np.array([obs[0]])

    def at_goal(self, obs, goal):
        return obs[0] >= goal[0]

    def desired_goal(self):
        return np.array([0.45])


class PickAndPlaceHindsightWrapper(HindsightWrapper):
    def __init__(self, env, default_reward=0):
        if isinstance(env, gym.Wrapper):
            assert isinstance(env.unwrapped, PickAndPlaceEnv)
            self.unwrapped_env = env.unwrapped
        else:
            assert isinstance(env, PickAndPlaceEnv)
            self.unwrapped_env = env
        super().__init__(env, default_reward)

    def achieved_goal(self, history):
        last_obs, = history[-1]
        return Goal(
            gripper=self.unwrapped_env.gripper_pos(last_obs),
            block=self.unwrapped_env.block_pos(last_obs))

    def at_goal(self, obs, goal, geofence=None):
        return any(self.unwrapped_env.at_goal(goal, o, geofence) for o in obs)

    def desired_goal(self):
        return self.unwrapped_env.goal()

    def vectorize_state(self, state):
        state = State(*state)
        state_history = list(map(np.concatenate, state.obs))
        return np.concatenate([np.concatenate(state_history), np.concatenate(state.goal)])


class ProgressiveWrapper(PickAndPlaceHindsightWrapper):
    def __init__(self, env: gym.Env, **kwargs):
        self.prev_block_pos = None
        self.time_step = 0
        self.max_time_step = 0
        self.surrogate_goal = None
        self.success_streak = 0
        self.max_success_streak = 1
        super().__init__(env, **kwargs)

    def step(self, action):
        s2, r, t, info = self.env.step(action)
        self.time_step += 1
        if self.surrogate_goal is None:
            goal = self.desired_goal()
            geofence = None
        else:
            goal = self.surrogate_goal
            geofence = .05
        new_s2 = State(obs=s2, goal=goal)
        at_goal = self.at_goal(s2, goal, geofence)
        if at_goal:
            self.success_streak += 1
        new_t = at_goal or t or self.time_step > self.max_time_step
        if new_t:
            if self.surrogate_goal is None:
                info['log count'] = {'successes': at_goal}
            self.time_step = 0
            if self.surrogate_goal is None and not at_goal:
                # if we failed on the main task
                self.surrogate_goal = self.achieved_goal(s2)
                block_joint = self.env.unwrapped.sim.jnt_qposadr('block1joint')
                self.prev_block_pos = (self.env.unwrapped.sim.qpos[block_joint + 3],
                                       self.env.unwrapped.sim.qpos[block_joint + 6])
            if self.success_streak == self.max_success_streak:
                # if we mastered the surrogate goal
                self.success_streak = 0
                self.max_time_step += 1
                self.surrogate_goal = None
                print('Mastered goal. Max time steps:', self.max_time_step)
        return new_s2, float(at_goal), new_t, info

    def reset(self):
        if self.success_streak == self.max_success_streak:
            self.success_streak = 0
        s1 = super().reset()
        if self.surrogate_goal is not None:
            block_joint = self.env.unwrapped.sim.jnt_qposadr('block1joint')
            self.env.unwrapped.sim.qpos[block_joint + 3] = self.prev_block_pos[0]
            self.env.unwrapped.sim.qpos[block_joint + 6] = self.prev_block_pos[1]
            self.env.unwrapped.sim.forward()
        return s1

