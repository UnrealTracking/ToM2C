# -*- coding: UTF-8 -*-
from MSMTC.DigitalPose2DBase.pose_env_base import Pose_Env_Base


class Gym:
    def make(self, env_id, render_save):
        reset_type = env_id.split('-v')[1]
        env = Pose_Env_Base(int(reset_type),render_save=render_save)
        return env


gym = Gym()
