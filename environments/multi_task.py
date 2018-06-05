from pathlib import Path

from gym import spaces
import numpy as np

from environments.pick_and_place import PickAndPlaceEnv, Goal


class MultiTaskEnv(PickAndPlaceEnv):
    def __init__(self, steps_per_action, geofence, min_lift_height):
        self._goal = None
        super().__init__(fixed_block=False,
                         steps_per_action=steps_per_action,
                         geofence=geofence,
                         min_lift_height=min_lift_height,
                         xml_filepath=Path('models', 'multi-task', 'world.xml'))
        self.env = self
        self.action_space = spaces.Box(
            low=np.array([-15, -20, -20, -10, -10]),
            high=np.array([35, 20, 20, 10, 10]),
            dtype=np.float32)
        self.lift_height = self._initial_block_pos[2] + geofence + min_lift_height

    def _set_new_goal(self):
        self._goal = np.random.uniform(low=[-0.161, -0.233, 0.457],
                                       high=[0.107, 0.201, 0.613])

    def goal(self):
        return Goal(gripper=self._goal, block=self._goal)

    def goal_3d(self):
        return self._goal

    def reset_qpos(self):
        self.init_qpos[[self.sim.jnt_qposadr('slide_x'),
                        self.sim.jnt_qposadr('slide_y'),
                        self.sim.jnt_qposadr('arm_flex_joint'),
                        self.sim.jnt_qposadr('wrist_roll_joint'),
                        self.sim.jnt_qposadr('hand_l_proximal_joint'),
                        ]] = np.random.uniform(low=[-.13, -.23, -1.1, -1.575, -0.0083],
                                               high=[.11, .25, .001, 1.575, 0.357])
        r = self.sim.jnt_qposadr('hand_r_proximal_joint')
        l = self.sim.jnt_qposadr('hand_l_proximal_joint')
        self.init_qpos[r] = self.init_qpos[l]

        block_joint = self.sim.jnt_qposadr('block1joint')
        self.init_qpos[[block_joint + 1,
                        block_joint + 3,
                        block_joint + 6]] = np.random.uniform(low=[-.23, 0, -1],
                                                              high=[.085, 1, 1])
        return self.init_qpos
