from environments.pick_and_place import PickAndPlaceEnv


class MultiTaskEnv(PickAndPlaceEnv):
    def _set_new_goal(self):
        raise NotImplementedError
