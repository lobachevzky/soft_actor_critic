import numpy as np
from gym import spaces

from environments.frozen_lake import FrozenLakeEnv
from environments.hindsight_wrapper import HindsightWrapper
from sac.utils import unwrap_env


class HierarchicalWrapper(HindsightWrapper):
    pass


class FrozenLakeHierarchicalWrapper(HierarchicalWrapper):
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

    @property
    def goal_space(self):
        fl = self.frozen_lake_env
        return spaces.Discrete(4 * (fl.nrow + fl.ncol))

    def get_direction(self, goal: int):
        fl = self.frozen_lake_env
        half = self.goal_space.n / 2
        offset = goal % half
        default_row = -fl.nrow if goal < half else fl.nrow
        default_col = fl.ncol if goal < half else -fl.ncol
        if offset < 2 * fl.ncol:  # offset describes col
            row_offset = default_row
            col_offset = offset - fl.ncol
        else:  # offset describes row
            row_offset = offset - 2 * fl.ncol - fl.nrow
            col_offset = default_col
        return np.array([row_offset, col_offset])
