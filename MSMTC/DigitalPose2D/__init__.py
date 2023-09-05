# -*- coding: UTF-8 -*-
from MSMTC.DigitalPose2D.pose_env_base import Pose_Env_Base


class Gym:
    def make(self, env_id, args):
        reset_type = env_id.split('-v')[1]
        env = Pose_Env_Base(int(reset_type),args)
        return env


gym = Gym()
