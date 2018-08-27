import math
import numpy as np

class SocialForceModel(object):
    # simulation parameters
    def __init__(self):
        # robot parameter
        self.max_speed = 2.0  # [m/s]
        self.min_speed = 0.0  # [m/s]
        self.max_yawrate = 1.0 # [rad/s]
        self.min_yawrate = -1.0 # [rad/s]
        self.pose = np.array([0.,0.])
        self.target = np.array([0.,0.])
    
    def reset(self,pose,target):
        self.pose = pose
        self.target = target

    def get_action(self):
        action = np.array([1.0,0.0])
        return action

