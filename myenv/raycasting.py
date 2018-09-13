# coding: utf-8

import numpy as np
import csv
from env_ex import MyEnv 

env = MyEnv()

def observe(pose):
    observation = np.zeros(env.NUM_LIDAR+env.NUM_TARGET)
    observation[0] = pose[0]
    observation[1] = pose[1]
    observation[2] = pose[2]
    #LIDAR
    for i in range(env.NUM_LIDAR):
        lidar = []
        for j in range(i*env.NUM_KERNEL,(i+1)*env.NUM_KERNEL-1):
            angle = j * (env.ANGLE_INCREMENT/env.NUM_KERNEL) - env.MAX_ANGLE
            lidar.append(env.raycasting(pose,angle))
        observation[i+3] = np.amin(lidar)
    return observation

with open('raycasting_test.csv', 'w') as f:
    writer = csv.writer(f,lineterminator='\n')
    data = []
    for i in range(10):#env.MAP_SIZE):
        x = i * env.MAP_RESOLUTION
        for j in range(10):#env.MAP_SIZE):
            y = j * env.MAP_RESOLUTION
            pose = np.array([x,y,0])
            if env.is_movable(pose):# and (not env.is_collision(pose)):
                for k in range(1):#360):
                    theta = k / 180 * np.pi
                    pose[2] = theta
                    data.append(observe(pose))
                    print(pose)
    writer.writerows(data)
