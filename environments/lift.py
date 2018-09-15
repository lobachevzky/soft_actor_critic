import numpy as np

from environments.hsr import HSREnv


class LiftEnv(HSREnv):
    def __init__(self, min_lift_height=.08, geofence=.05, **kwargs):
        self.min_lift_height = min_lift_height + geofence
        super().__init__(geofence=geofence, **kwargs)

    def reset(self):
        obs = super().reset()
        self.goal = self.block_pos() + np.array([0, 0, self.min_lift_height])
        return obs

    def _is_successful(self):
        return self.block_pos()[2] > self.initial_block_pos[2] + self.min_lift_height
