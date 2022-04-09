# -*- coding: UTF-8 -*-
from MSMTC.DigitalPose2D.pose_env_base import Pose_Env_Base


class Gym:
    def make(self, env_id, render_save, num_agents=-1,num_targets=-1):
        reset_type = env_id.split('-v')[1]
        env = Pose_Env_Base(int(reset_type),render_save=render_save, num_agents=num_agents,num_targets=num_targets)
        return env


gym = Gym()
