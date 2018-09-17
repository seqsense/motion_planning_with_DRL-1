import math
import numpy as np

class SocialForceModel(object):
    # simulation parameters
    def __init__(self,radius):
        # robot parameter
        self.avg_speed = 0.5  # [m/s]
        self.alpha = 1.0
        self.beta = 0.0
        self.B = 5.0
        self.C = 0.0
        self.radius = radius
        self.FOV = np.pi * 0.4

        self.pose = np.array([0.,0.])
        self.target = np.array([0.,0.])
        self.action = np.array([0.,0.])
    
    def reset(self,pose,target):
        self.pose = pose
        self.target = target

    def calc_dis(self,a,b):
        dis = np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
        return dis

    def abs_vector(self,a):
        return np.sqrt(a[0]**2+a[1]**2)

    def get_action(self,pose,ob):
        if self.calc_dis(pose,self.target) < self.radius:
            e = 0.0
            return np.zeros(2)
        else:
            e = (self.target-pose)/self.calc_dis(pose,self.target)
        v_a = (e*self.avg_speed)# - self.action
        n = -np.array([ob[0]-pose[0],ob[1]-pose[1]])
        n = n / self.abs_vector(n)
        v_b = (np.exp(self.radius*2-self.calc_dis(pose,ob)/self.B)) * n 
        if np.dot(e,n) > self.abs_vector(n)*np.cos(self.FOV):
            v_b *= self.C
        #print(v_a, v_b)
        self.action = self.alpha * v_a + self.beta * v_b
        
        return self.action

