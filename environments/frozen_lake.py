import sys
import time
from typing import Tuple

import numpy as np
from gym import utils
from gym.envs.toy_text.discrete import DiscreteEnv
from gym.envs.toy_text.frozen_lake import MAPS
from gym.spaces import Box
from six import StringIO

from environments.multi_task import Observation

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


def random_walk(m: np.ndarray, pos: tuple, n_steps: int):
    explored = []
    while n_steps > 0:
        explored.append(pos)
        low = np.zeros(2)
        high = np.shape(m)
        apos = np.array(pos)
        next_positions = [
            apos + d for d in DIRECTIONS if np.all((low <= (apos + d)) * (
                (apos + d) < high)) and not tuple(apos + d) in explored
        ]
        if not next_positions:
            return pos
        pos = tuple(next_positions[np.random.randint(len(next_positions))])
        n_steps -= 1
    return pos


class FrozenLakeEnv(DiscreteEnv):
    def __init__(self,
                 map_dims: Tuple[int, int] = (4, 4),
                 slip_prob: float = 0,
                 random_start: bool = False,
                 random_goal: bool = False,
                 random_map=False):

        self.slip_prob = slip_prob
        self.random_start = random_start
        self.random_goal = random_goal
        h, w = map_dims
        self.n_state = h * w
        self.n_row = h
        self.n_col = w
        self.start = (0, 0)
        self.goal = (h - 1, w - 1)
        self.goal_vector = self.one_hotify(h * w - 1)

        if random_map:
            while True:
                self.map = np.random.choice([b'F', b'H'], [h, w], p=[.85, .15])
                if random_walk(self.map, self.start, 1) != self.start:
                    break  # start is not surrounded by holes
            self.original_map = np.copy(self.map)
        else:
            self.map = np.array([list(r) for r in MAPS[f"{h}x{w}"]])
            self.map = np.asarray(self.map, dtype='c')
            self.original_map = np.copy(self.map)
            self.original_map[0, 0] = b'F'
            self.original_map[-1, -1] = b'F'

        self.map[self.start] = b'S'
        self.map[self.goal] = b'G'
        super().__init__(
            nS=self.n_state,
            nA=4,
            P=(self.get_transitions(self.map)),
            isd=(self.get_initial_state_distribution(self.map)))

        if self.random_goal:
            size_obs = self.observation_space.n * 2
        else:
            size_obs = self.observation_space.n
        self.observation_space = Box(low=np.zeros(size_obs), high=np.ones(size_obs))

    def reset(self):
        time.sleep(1)
        if self.random_start:
            self.start = np.random.randint(self.n_row), np.random.randint(self.n_col)
        if self.random_goal:
            self.goal = random_walk(
                self.map, self.start, n_steps=self.n_row * self.n_col // 2)
            if self.goal == self.start:
                # Can only happen is starts are random. Just roll again.
                return self.reset()

        if self.random_start or self.random_goal:
            self.map = np.copy(self.original_map)
            self.map[self.start] = b'S'
            self.map[self.goal] = b'G'
            h, w = self.goal
            self.goal_vector = self.one_hotify(h * self.n_col + w)
            self.isd = self.get_initial_state_distribution(self.map)
            self.P = self.get_transitions(self.map)
        if self.random_goal:
            observation = Observation(
                observation=self.one_hotify(super().reset()), goal=self.goal_vector)
            return observation
        else:
            return self.one_hotify(super().reset())

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

    @staticmethod
    def get_initial_state_distribution(desc):
        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()
        return isd

    def get_transitions(self, desc):
        P = {s: {a: [] for a in range(4)} for s in range(self.n_state)}

        def to_s(row, col):
            return row * self.n_col + col

        def inc(row, col, a):
            if a == 0:  # left
                col = max(col - 1, 0)
            elif a == 1:  # down
                row = min(row + 1, self.n_row - 1)
            elif a == 2:  # right
                col = min(col + 1, self.n_col - 1)
            elif a == 3:  # up
                row = max(row - 1, 0)
            return row, col

        for row in range(self.n_row):
            for col in range(self.n_col):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if self.slip_prob:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = float(newletter == b'G')
                                li.append((self.slip_prob, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done))
        return P

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.n_col, self.s % self.n_col
        desc = self.map.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right",
                                             "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile
