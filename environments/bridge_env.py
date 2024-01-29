import sys
from time import sleep
from panda-gym.panda_gym.envs.core_multi_robot import RobotTaskEnv
from panda-gym.panda_gym.pybullet import PyBullet
import numpy as np
from bridge_task import MyTask
from panda-gym.panda_gym.envs.robots.panda import Panda
class MyRobotTaskEnv(RobotTaskEnv):
    """My robot-task environment."""

    def __init__(self, render_mode="rgb_array"):
        sim = PyBullet(render_mode=render_mode)
        robots = [Panda(sim,base_position=np.array([0,0,0]),base_orientation=np.array([0,0,0]),body_name = "")]
        task = MyTask(sim)
        super().__init__(robots, task)

