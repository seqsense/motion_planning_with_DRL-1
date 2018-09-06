# coding: utf-8

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

def make_circle(MAP,cx,cy,r):
    for i in range(cx-r-1,cx+r):
        for j in range(cy-r-1,cy+r):
            if np.sqrt((i-cx)**2+(j-cy)**2) < r:
                MAP[i][j] = 1
 
def make_rectangle(MAP,l,r,t,b):
    for i in range(l-1,r):
        for j in range(t-1,b):
            MAP[i][j] = 1

def reset_map(size):
    MAP = np.zeros((size,size))
    for i in range(0,size-1):
        MAP[i][0] = 1
        MAP[i][size-1] = 1
    for j in range(1,size-2):
        MAP[0][j] = 1
        MAP[size-1][j] = 1
    make_rectangle(MAP,350,650,600,650)
    make_rectangle(MAP,350,650,350,400)
    make_rectangle(MAP,150,200,375,625)
    make_rectangle(MAP,800,850,375,625)
    return MAP

def angle_nomalize(z):
    return np.arctan2(np.sin(z),np.cos(z))

def angle_diff(a,b):
    a = angle_nomalize(a)
    b = angle_nomalize(b)
    d1 = a -b
    d2 = 2.0 * np.pi - abs(d1)
    if d1 > 0.0:
        d2 *= -1.0
    if abs(d1) < abs(d2):
        return d1
    else:
        return d2

class MyEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.MAP_SIZE = 1000
        self.MAP_RESOLUTION = 0.01
        #self.MAP,self.l1, self.l2 = reset_map(self.MAP_SIZE)
        self.MAP = reset_map(self.MAP_SIZE)
        self.WORLD_SIZE = self.MAP_SIZE * self.MAP_RESOLUTION
        self.DT = 0.1 #seconds between state updates

        self.robot_radius = 0.30 #[m]

        #action
        self.max_linear_velocity = 1.0
        self.min_linear_velocity = 0.0
        self.max_angular_velocity = 1.0
        self.min_angular_velocity = -1.0
        self.max_accel = 1.5
        self.max_dyawrate = 3.0

        self.action_low = np.array([self.min_linear_velocity,self.min_angular_velocity])
        self.action_high = np.array([self.max_linear_velocity,self.max_angular_velocity])

        self.action_space = spaces.Box(self.action_low, self.action_high, dtype = np.float32)
        
        #observation
        self.min_range = 0.0
        self.max_range = 10.0
        self.min_distance = 0.0
        self.max_distance = np.sqrt(2) * self.WORLD_SIZE
        self.NUM_LIDAR = 36
        self.NUM_KERNEL = 20
        self.NUM_TARGET = 3 
        self.MAX_ANGLE = 0.5*np.pi
        self.ANGLE_INCREMENT = self.MAX_ANGLE * 2.0 / self.NUM_LIDAR
        self.RANGE_MAX = 10
        self.observation_low = np.full(self.NUM_LIDAR+self.NUM_TARGET, self.min_range)
        self.observation_low[self.NUM_LIDAR] = self.min_distance
        self.observation_low[self.NUM_LIDAR+1] = -1.
        self.observation_low[self.NUM_LIDAR+2] = -1.
        self.observation_high = np.full(self.NUM_LIDAR+self.NUM_TARGET, self.max_range)
        self.observation_high[self.NUM_LIDAR] = self.max_distance
        self.observation_high[self.NUM_LIDAR+1] = 1.
        self.observation_high[self.NUM_LIDAR+2] = 1.
        self.observation_space = spaces.Box(self.observation_low, self.observation_high, dtype = np.float32)
        
        self.viewer = None
        self.seed()
        self.reset()

    def reset(self):
        theta = np.random.rand()*2.0*np.pi
        self.pose = np.array([self.WORLD_SIZE*0.5, self.WORLD_SIZE*0.5, theta])
        while True:
            self.target = np.array([np.random.rand()*self.WORLD_SIZE, np.random.rand()*self.WORLD_SIZE,0.0])
            if not self.is_collision(self.target):
                break
        self.MAP = reset_map(self.MAP_SIZE) 
        self.dis = np.sqrt((self.target[0]-self.pose[0])**2 + (self.target[1]-self.pose[1])**2)
        self.pre_dis = self.dis
        self.observation = self.observe()
        self.done = False
        return self.observation

    def step(self, action):
        #pose update
        self.pose[0] = self.pose[0] + action[0]*np.cos(self.pose[2])*self.DT
        self.pose[1] = self.pose[1] + action[0]*np.sin(self.pose[2])*self.DT
        self.pose[2] = self.pose[2] + action[1]*self.DT
        self.pose[2] %= 2.0 * np.pi
                
        self.observation = self.observe()
        reward = self.get_reward()
        self.done = self.is_done()
        return self.observation, reward, self.done, {}

    def render(self, mode='human', close=False):
        screen_width = 600
        screen_height = 600
        margin = 0.2
        world_width = self.WORLD_SIZE + margin * 2.0
        scale = screen_width / world_width
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width,screen_height)
            #wall
            l = margin * scale
            r = (margin + self.WORLD_SIZE) *scale
            t = margin * scale
            b = (margin + self.WORLD_SIZE) *scale
            wall = rendering.PolyLine([(l,b),(l,t),(r,t),(r,b)],True)
            wall.set_color(0.,0.,0.)
            self.viewer.add_geom(wall)
            #robot
            robot = rendering.make_circle(self.robot_radius*scale)
            self.robot_trans = rendering.Transform()
            robot.add_attr(self.robot_trans)
            robot.set_color(0.0,0.0,1.0)
            self.viewer.add_geom(robot)
            robot_orientation = rendering.make_capsule(self.robot_radius*scale,1.0)
            self.orientation_trans = rendering.Transform()
            robot_orientation.set_color(0.0,1.0,0.0)
            robot_orientation.add_attr(self.orientation_trans)
            self.viewer.add_geom(robot_orientation)
            #target
            target = rendering.make_circle(self.robot_radius*0.3*scale)
            self.target_trans = rendering.Transform()
            target.add_attr(self.target_trans)
            target.set_color(1.0,0.0,0.0)
            self.viewer.add_geom(target)
            #obstract
            l = (margin+350*self.MAP_RESOLUTION) * scale
            r = (margin+650*self.MAP_RESOLUTION) * scale
            t = (margin+350*self.MAP_RESOLUTION) * scale
            b = (margin+400*self.MAP_RESOLUTION) * scale
            ob1 = rendering.FilledPolygon([(l,b),(l,t),(r,t),(r,b)])
            ob1.set_color(0.,0.,0.)
            self.viewer.add_geom(ob1)
            l = (margin+350*self.MAP_RESOLUTION) * scale
            r = (margin+650*self.MAP_RESOLUTION) * scale
            t = (margin+600*self.MAP_RESOLUTION) * scale
            b = (margin+650*self.MAP_RESOLUTION) * scale
            ob3 = rendering.FilledPolygon([(l,b),(l,t),(r,t),(r,b)])
            ob3.set_color(0.,0.,0.)
            self.viewer.add_geom(ob3)
            
            l = (margin+150*self.MAP_RESOLUTION) * scale
            r = (margin+200*self.MAP_RESOLUTION) * scale
            t = (margin+375*self.MAP_RESOLUTION) * scale
            b = (margin+625*self.MAP_RESOLUTION) * scale
            ob4 = rendering.FilledPolygon([(l,b),(l,t),(r,t),(r,b)])
            ob4.set_color(0.,0.,0.)
            self.viewer.add_geom(ob4)
            l = (margin+800*self.MAP_RESOLUTION) * scale
            r = (margin+850*self.MAP_RESOLUTION) * scale
            t = (margin+375*self.MAP_RESOLUTION) * scale
            b = (margin+625*self.MAP_RESOLUTION) * scale
            ob5 = rendering.FilledPolygon([(l,b),(l,t),(r,t),(r,b)])
            ob5.set_color(0.,0.,0.)
            self.viewer.add_geom(ob5)

        robot_x = (margin + self.pose[0]) * scale
        robot_y = (margin + self.pose[1]) * scale
        robot_orientation = self.pose[2]
        self.robot_trans.set_translation(robot_x, robot_y) 
        self.orientation_trans.set_translation(robot_x,robot_y)
        self.orientation_trans.set_rotation(robot_orientation)
        self.target_trans.set_translation((self.target[0]+margin)*scale,(self.target[1]+margin)*scale)
        from gym.envs.classic_control import rendering
        #lidar
        for i in range(self.NUM_LIDAR):
            lidar = rendering.make_capsule(scale*self.observation[i],1.0)
            lidar_trans = rendering.Transform()
            lidar_trans.set_translation(robot_x,robot_y)
            lidar_trans.set_rotation(self.pose[2] + i*self.ANGLE_INCREMENT - self.MAX_ANGLE)
            lidar.set_color(1.0,0.0,0.0)
            lidar.add_attr(lidar_trans)
            self.viewer.add_onetime(lidar)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
                
    def get_reward(self):
        reward = 0
        self.dis = np.sqrt((self.target[0]-self.pose[0])**2 + (self.target[1]-self.pose[1])**2)
        if self.is_goal():
            reward = 5.
        elif (not self.is_movable(self.pose)) or self.is_collision(self.pose):
            reward = -5.
        else:
            reward = (self.pre_dis-self.dis)*0.15
        if abs(self.pre_dis-self.dis) < 1e-6:
            reward -=0.05
        self.pre_dis = self.dis
#        reward += 1./(200.*np.pi)*angle_diff(np.arctan2((self.pose[1]-self.target[1]),(self.pose[0]-self.target[0])),self.pose[2])
        return reward

    def is_movable(self, pose):
        i = int(pose[0]/self.MAP_RESOLUTION)
        j = int(pose[1]/self.MAP_RESOLUTION)
        return (0 <= pose[0] < self.WORLD_SIZE and 0 <= pose[1] < self.WORLD_SIZE and self.MAP[i][j] == 0)

    def is_collision(self, pose):
        min_dis = self.RANGE_MAX
        margin_collision = self.robot_radius * 0.1
        NUM_CHECK = 36
        for i in range(NUM_CHECK):
            angle = i * (2.0 * np.pi / NUM_CHECK)
            dis = self.raycasting(pose,angle)
            if min_dis > dis:
                min_dis = dis
        return min_dis < self.robot_radius + margin_collision
    
    def is_goal(self):
        return self.dis < self.robot_radius

    def observe(self):
        observation = np.zeros(self.observation_space.shape[0])
        #LIDAR
        for i in range(self.NUM_LIDAR):
            lidar = []
            for j in range(i*self.NUM_KERNEL,(i+1)*self.NUM_KERNEL):
                angle = j * (self.ANGLE_INCREMENT/self.NUM_KERNEL) - self.MAX_ANGLE
                lidar.append(self.raycasting(self.pose,angle))
            observation[i] = np.amin(lidar)
        #pose
        observation[self.NUM_LIDAR] = np.sqrt((self.target[0]-self.pose[0])**2 +(self.target[1]-self.pose[1])**2)
        theta = np.arctan2((self.target[1]-self.pose[1]),(self.target[0]-self.pose[0]))
        theta = angle_diff(theta,self.pose[2])
        observation[self.NUM_LIDAR+1] = np.sin(theta)
        observation[self.NUM_LIDAR+2] = np.cos(theta)
        return observation

    def is_done(self):
        return (not self.is_movable(self.pose)) or self.is_collision(self.pose) or self.is_goal()

    def raycasting(self,pose, angle):
        x0 = int(pose[0]/self.MAP_RESOLUTION)
        y0 = int(pose[1]/self.MAP_RESOLUTION)
        x1 = int((pose[0]+self.RANGE_MAX * np.cos(pose[2]+angle))/self.MAP_RESOLUTION)
        y1 = int((pose[1]+self.RANGE_MAX * np.sin(pose[2]+angle))/self.MAP_RESOLUTION)
        steep = False
        if abs(y1-y0) > abs(x1-x0):
            steep = True
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        dx, dy = abs(x1-x0), abs(y1-y0)
        error, derror = 0, dy
        x, y = x0, y0
        x_step, y_step = -1, -1
        if x0<x1:
            x_step = 1
        if y0<y1:
            y_step = 1
        if steep:
            pose_ = np.array([y*self.MAP_RESOLUTION,x*self.MAP_RESOLUTION,0])
            if not self.is_movable(pose_):
                return np.sqrt((x-x0)**2 + (y-y0)**2) * self.MAP_RESOLUTION
        else:
            pose_ = np.array([x*self.MAP_RESOLUTION,y*self.MAP_RESOLUTION,0])
            if not self.is_movable(pose_):
                return np.sqrt((x-x0)**2 + (y-y0)**2) * self.MAP_RESOLUTION
        while x != (x1 + x_step):
            x = x + x_step
            error = error + derror
            if 2.0*error >= dx:
                y = y + y_step
                error = error - dx
                if steep:
                    pose_ = np.array([y*self.MAP_RESOLUTION,x*self.MAP_RESOLUTION,0])
                    if not self.is_movable(pose_):
                        return np.sqrt((x-x0)**2 + (y-y0)**2) * self.MAP_RESOLUTION
                else:
                    pose_ = np.array([x*self.MAP_RESOLUTION,y*self.MAP_RESOLUTION,0])
                    if not self.is_movable(pose_):
                        return np.sqrt((x-x0)**2 + (y-y0)**2) * self.MAP_RESOLUTION
                
        return self.RANGE_MAX
