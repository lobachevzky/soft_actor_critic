import numpy as np
from gym import spaces

from environments.hindsight_wrapper import HindsightWrapper, FrozenLakeHindsightWrapper


class HierarchicalWrapper(HindsightWrapper):
    pass


class FrozenLakeHierarchicalWrapper(HierarchicalWrapper, FrozenLakeHindsightWrapper):
    def get_direction(self, goal: int):
        fl = self.frozen_lake_env
        half = 2 * (fl.nrow + fl.ncol)
        offset = goal % half
        default_row = -fl.nrow if goal < half else fl.nrow
        default_col = fl.ncol if goal < half else -fl.ncol
        if offset < 2 * fl.ncol:  # offset describes col
            row_offset = default_row
            col_offset = offset - fl.ncol
        else:  # offset describes row
            row_offset = offset - 2 * fl.ncol - fl.nrow
            col_offset = default_col
        direction = np.array([row_offset, col_offset], dtype=float)
        return direction / np.linalg.norm(direction)
