import math
import numpy as np

class SocialForceModel(object):
    # simulation parameters
    def __init__(self,radius):
        # robot parameter
        self.avg_speed = 1.0  # [m/s]
        self.alpha = 1.0
        self.beta = 0.5
        self.dt = 0.01
        self.B = 5.0
        self.C = 0.3
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

    def get_action(self,pose,ob1,ob2,ob3):
        if self.calc_dis(pose,self.target) < self.radius:
            e = 0.0
            return np.zeros(2)
        else:
            e = (self.target-pose)/self.calc_dis(pose,self.target)
        v_a = (e*self.avg_speed)# - self.action
        n1 = -np.array([ob1[0]-pose[0],ob1[1]-pose[1]])
        n1 = n1 / self.abs_vector(n1)
        v_b1 = (np.exp(self.radius*2-self.calc_dis(pose,ob1)/self.B)) * n1 
        if np.dot(e,n1) < self.abs_vector(n1)*np.cos(self.FOV):
            v_b1 *= self.C
        n2 = -np.array([ob2[0]-pose[0],ob2[1]-pose[1]])
        n2 = n2 / self.abs_vector(n2)
        v_b2 = (np.exp(self.radius*2-self.calc_dis(pose,ob2)/self.B)) * n2 
        if np.dot(e,n2) < self.abs_vector(n2)*np.cos(self.FOV):
            v_b2 *= self.C
        n3 = -np.array([ob3[0]-pose[0],ob3[1]-pose[1]])
        n3 = n3 / self.abs_vector(n3)
        v_b3 = (np.exp(self.radius*2-self.calc_dis(pose,ob3)/self.B)) * n3
        if np.dot(e,n3) < self.abs_vector(n3)*np.cos(self.FOV):
            v_b3 *= self.C
        #print(v_a, v_b)
        v_b = v_b1 + v_b2 + v_b3
        self.action = self.alpha * v_a + self.beta * v_b
        
        return self.action

