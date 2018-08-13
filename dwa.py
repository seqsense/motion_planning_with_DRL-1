import math
import numpy as np
import gym
import myenv

env = gym.make('myenv-v2')
class Config():
    # simulation parameters
    def __init__(self):
        # robot parameter
        self.max_speed = env.max_linear_velocity  # [m/s]
        self.min_speed = env.min_linear_velocity  # [m/s]
        self.max_yawrate = env.max_angular_velocity  # [rad/s]
        self.min_yawrate = env.min_angular_velocity  # [rad/s]
        self.max_accel = env.max_accel  # [m/ss]
        self.max_dyawrate = env.max_dyawrate  # [rad/ss]
        self.v_reso = 0.01  # [m/s]
        self.yawrate_reso = 0.1# * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s]
        self.predict_time = 5.0  # [s]
        self.to_goal_cost_gain = 100
        self.vel_cost_gain = 0.
        self.yaw_cost_gain = 0. 
        self.robot_radius = env.robot_radius  # [m]

def motion(x, u, dt):
    # motion model

    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[2] += u[1] * dt
    x[2] %= 2.0 * np.pi
    return x

def calc_dynamic_window(u, config):

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          config.min_yawrate, config.max_yawrate]

    # Dynamic window from motion model
    Vd = [u[0] - config.max_accel * config.dt,
          u[0] + config.max_accel * config.dt,
          u[1] - config.max_dyawrate * config.dt,
          u[1] + config.max_dyawrate * config.dt]
    #  print(Vs, Vd)

    #  [vmin,vmax, yawrate min, yawrate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    #print(dw)

    return dw 


def calc_trajectory(v, y, config):

    x = np.zeros(3)
    traj = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt

    #  print(len(traj))
    return traj


def calc_final_input(u, dw, config, goal, ob):

    min_cost = 10000.0
    costs = [0,0,0]
    min_u = u
    min_u[0] = 0.0

    # evalucate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_reso):
        for y in np.arange(dw[2], dw[3], config.yawrate_reso):
            traj = calc_trajectory(v, y, config)

            # calc cost
            to_goal_cost = calc_to_goal_cost(traj, goal, config)
            ob_cost = calc_obstacle_cost(traj, ob, config)
            v_cost = calc_vel_cost(v,y,config)
            #print(to_goal_cost,ob_cost)

            final_cost = to_goal_cost + ob_cost + v_cost
            #print(v,y,final_cost)
            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                costs = [to_goal_cost,ob_cost,v_cost]
                min_u = [v, y]

    return min_u,min_cost,costs


def calc_obstacle_cost(traj, ob, config):
    # calc obstacle cost inf: collistion, 0:free

    minr = float("Inf")
    for i in range(env.NUM_LIDAR):
        oz = (2.0*i/env.NUM_LIDAR-1.0)*(np.pi/2.0)
        ox = ob[i]*np.cos(oz)
        oy = ob[i]*np.sin(oz)
        
        for j in range(len(traj[:,1])):
            dx = traj[j, 0] - ox
            dy = traj[j, 1] - oy
            r = np.sqrt(dx**2 + dy**2)
            if r <= config.robot_radius:
                return float("Inf")  # collisiton

            if minr >= r:
                minr = r

    return 1.0 / minr  # OK


def calc_to_goal_cost(traj, goal, config):
    # calc to goal cost. It is 2D norm.

    dy = goal[0] - traj[-1, 0]
    dx = goal[1] - traj[-1, 1]
    goal_dis = math.sqrt(dx**2 + dy**2)
    cost = config.to_goal_cost_gain * goal_dis

    return cost

def calc_vel_cost(v,y,config):
    return config.vel_cost_gain * v + config.yaw_cost_gain * abs(y)

def dwa_control(u, config, state):
    #print(state)
        # Dynamic Window control
    lidar = np.zeros(env.NUM_LIDAR)
    for i in range(env.NUM_LIDAR):
        lidar[i] = state[i]
    #print(lidar)
    goal = np.array([state[env.NUM_LIDAR]*state[env.NUM_LIDAR+2], state[env.NUM_LIDAR]*state[env.NUM_LIDAR+1]])
    #print(goal)
    dw = calc_dynamic_window(u, config)
    #print(dw)
    u,min_cost,costs = calc_final_input(u, dw, config, goal,lidar)
    #print(min_cost,costs)
    return u

def main():
    
    for episode in range(100):
        state = env.reset()
        done = False
        ep_r = 0
    
        action = np.array([0.0, 0.0])
        config = Config()

        for t in range(1000):
            env.render()
            
            action_ = dwa_control(action, config, state)
            #print(action_)
            state_ ,reward, done, info = env.step(action_)
            if t == 1000-1: done = True
            ep_r += reward
            action = action_
            state = state_
            #print(reward)
            if done:
                print("Episode %d finished after %d timesteps, reward: %f "%(episode+1,t+1,ep_r))
                break

if __name__ == '__main__':
    main()
