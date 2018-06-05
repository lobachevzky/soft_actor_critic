from pathlib import Path

from gym import spaces
import numpy as np

from environments.pick_and_place import PickAndPlaceEnv, Goal


class MultiTaskEnv(PickAndPlaceEnv):
    def __init__(self, steps_per_action, geofence):
        self._goal = None
        super().__init__(fixed_block=False,
                         steps_per_action=steps_per_action,
                         geofence=geofence,
                         xml_filepath=Path('models', 'multi-task', 'world.xml'))
        self.env = self
        self.action_space = spaces.Box(
            low=np.array([-15, -20, -20, -10, -10]),
            high=np.array([35, 20, 20, 10, 10]),
            dtype=np.float32)

    def _set_new_goal(self):
        self._goal = np.random.uniform(low=[-0.161, - 0.233, 0.457],
                                       high=[0.107, 0.201, 0.613])

    def goal(self):
        return Goal(gripper=self._goal, block=self._goal)

    def goal_3d(self):
        return self._goal
