import sys
import time
from typing import Tuple

import numpy as np
from gym import utils
from gym.envs.toy_text.discrete import DiscreteEnv
import gym.envs.toy_text.frozen_lake
from gym.spaces import Box
from six import StringIO

from environments.multi_task import Observation

MAPS = gym.envs.toy_text.frozen_lake.MAPS
MAPS["3x3"] = [
    "SFF",
    "FHF",
    "FFG",
]

MAPS["3x4"] = ["SFFF", "FHFH", "HFFG"]

DIRECTIONS = np.array([
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
])

def in_bounds(pos, n_row, n_col):
    pos = np.array(pos)
    lower = np.zeros(2)
    upper = np.array([n_row, n_col])
    return np.all((lower <= pos) * (pos < upper))

def random_walk(m: np.ndarray, pos: tuple, n_steps: int):
    explored = []
    while n_steps > 0:
        explored.append(pos)
        low = np.zeros(2)
        high = np.shape(m)
        apos = np.array(pos)
        next_positions = [
            apos + d for d in DIRECTIONS if in_bounds(apos + d, *high)
                and not tuple(apos + d) in explored
            ]
        if not next_positions:
            return pos
        pos = tuple(next_positions[np.random.randint(len(next_positions))])
        n_steps -= 1
    return pos


class FrozenLakeEnv(gym.envs.toy_text.frozen_lake.FrozenLakeEnv):
    def __init__(self,
                 map_dims: Tuple[int, int] = (4, 4),
                 is_slippery: bool = False,
                 random_start: bool = False,
                 random_goal: bool = False,
                 random_map=False):

        self.is_slippery = is_slippery
        self.random_start = random_start
        self.random_goal = random_goal
        h, w = map_dims
        self.n_state = h * w
        self.n_row = h
        self.n_col = w
        self.start = (0, 0)
        self.goal = (h - 1, w - 1)
        self.goal_vector = self.one_hotify(self.to_s(h - 1, w - 1))

        if random_map:
            while True:
                desc = np.random.choice([b'F', b'H'], [h, w], p=[.85, .15])
                if random_walk(desc, self.start, 1) != self.start:
                    break  # start is not surrounded by holes
            self.original_desc = np.copy(desc)
        else:
            desc = np.array([list(r) for r in MAPS[f"{h}x{w}"]])
            self.original_desc = np.copy(desc)
            self.original_desc[0, 0] = b'F'
            self.original_desc[-1, -1] = b'F'

        desc[self.start] = b'S'
        desc[self.goal] = b'G'
        super().__init__(desc=desc, is_slippery=is_slippery)

        if self.random_goal:
            size_obs = self.observation_space.n * 2
        else:
            size_obs = self.observation_space.n
        self.observation_space = Box(low=np.zeros(size_obs), high=np.ones(size_obs))

    def inc(self, row, col, a):
        if a == 0:  # left
            col = max(col - 1, 0)
        elif a == 1:  # down
            row = min(row + 1, self.n_row - 1)
        elif a == 2:  # right
            col = min(col + 1, self.n_col - 1)
        elif a == 3:  # up
            row = max(row - 1, 0)
        return row, col

    def set_transition(self, pos: tuple):
        pos = tuple(pos)
        s = self.to_s(*pos)
        for a in range(4):
            letter = self.desc[pos]
            if letter in b'GH':
                self.P[s][a] = [(1.0, s, 0, True)]
            else:
                if self.is_slippery:
                    for b in [(a - 1) % 4, a, (a + 1) % 4]:
                        newrow, newcol = self.inc(*pos, b)
                        newstate = self.to_s(newrow, newcol)
                        newletter = self.desc[newrow, newcol]
                        done = bytes(newletter) in b'GH'
                        rew = float(newletter == b'G')
                        self.P[s][a] = [(1.0 / 3.0, newstate, rew, done)]
                else:
                    newrow, newcol = self.inc(*pos, a)
                    newstate = self.to_s(newrow, newcol)
                    newletter = self.desc[newrow, newcol]
                    done = bytes(newletter) in b'GH'
                    rew = float(newletter == b'G')
                    self.P[s][a] = [(1.0, newstate, rew, done)]

    def mutate_desc(self, old_pos, new_pos):
        letter = self.desc[old_pos]
        self.desc[old_pos] = self.original_desc[old_pos]
        self.desc[new_pos] = letter

    def reset(self):
        time.sleep(1)
        if self.random_start:
            old_start = self.start
            while True:
                self.start = np.random.randint(self.n_row), np.random.randint(self.n_col)
                if self.desc[self.start] != b'G':
                    break
            self.mutate_desc(old_start, self.start)
            self.isd[self.to_s(*old_start)] = 0
            self.isd[self.to_s(*self.start)] = 1
        if self.random_goal:
            old_goal = self.goal
            new_goal = random_walk(
                self.desc, self.start, n_steps=self.n_row * self.n_col // 2)
            if new_goal == self.start:
                # Can only happen if starts or goals are random. Just roll again.
                return self.reset()
            self.mutate_desc(old_goal, new_goal)
            self.set_transition(new_goal)
            self.set_transition(old_goal)
            for d in DIRECTIONS:
                pos = np.array(new_goal) + d
                if in_bounds(pos, self.n_col, self.n_row):
                    self.set_transition(pos)
            import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
            self.goal = new_goal

        if self.random_goal:
            observation = Observation(
                observation=self.one_hotify(super().reset()), goal=self.goal_vector)
            return observation
        else:
            return self.one_hotify(super().reset())

    def to_s(self, row, col):
        return row * self.n_col + col

    def step(self, a):
        s, r, t, i = super().step(a)
        s = self.one_hotify(s)
        if self.random_goal:
            s = Observation(observation=s, goal=self.goal_vector)
        return s, r, t, i

    def one_hotify(self, s):
        array = np.zeros(self.n_state)
        array[s] = 1
        return array

    def render(self, *args, **kwargs):
        time.sleep(.5)
        return super().render(*args, **kwargs)
