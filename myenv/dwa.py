import math
import numpy as np
import gym
import myenv

env = gym.make('myenv-v0')
class Expert():
    # simulation parameters
    def __init__(self):
        # robot parameter
        self.max_speed = 1.0#env.max_linear_velocity  # [m/s]
        self.min_speed = 0.0#env.min_linear_velocity  # [m/s]
        self.max_yawrate = 1.0#env.max_angular_velocity  # [rad/s]
        self.min_yawrate = -1.0#env.min_angular_velocity  # [rad/s]
        self.max_accel = 1.5#env.max_accel  # [m/ss]
        self.max_dyawrate = 3.0#env.max_dyawrate  # [rad/ss]
        self.v_reso = 0.01  # [m/s]
        self.yawrate_reso = 0.1  # [rad/s]
        self.dt = 0.1  # [s]
        self.predict_time = 5.0  # [s]
        self.to_goal_cost_gain = 50
        self.vel_cost_gain = 0.
        self.yaw_cost_gain = 0.
        self.robot_radius = 0.2#env.robot_radius  # [m]

    def motion(self,x, u, dt):
        # motion model

        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[2] += u[1] * dt
        x[2] %= 2.0 * np.pi
        return x

    def calc_dynamic_window(self,u):

        # Dynamic window from robot specification
        Vs = [self.min_speed, self.max_speed,
              self.min_yawrate, self.max_yawrate]

        # Dynamic window from motion model
        Vd = [u[0] - self.max_accel * self.dt,
              u[0] + self.max_accel * self.dt,
              u[1] - self.max_dyawrate * self.dt,
              u[1] + self.max_dyawrate * self.dt]
        #  print(Vs, Vd)

        #  [vmin,vmax, yawrate min, yawrate max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        #print(dw)

        return dw


    def calc_trajectory(self,v, y):

        x = np.zeros(3)
        traj = np.array(x)
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, y], self.dt)
            traj = np.vstack((traj, x))
            time += self.dt

        #  print(len(traj))
        return traj


    def calc_final_input(self,u, dw, goal, ob):

        min_cost = 10000.0
        costs = [0,0,0]
        min_u = u
        min_u[0] = 0.0

        # evalucate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1], self.v_reso):
            for y in np.arange(dw[2], dw[3], self.yawrate_reso):
                traj = self.calc_trajectory(v, y)

                # calc cost
                to_goal_cost = self.calc_to_goal_cost(traj, goal)
                ob_cost = self.calc_obstacle_cost(traj, ob)
                v_cost = self.calc_vel_cost(v,y)
                #print(to_goal_cost,ob_cost)

                final_cost = to_goal_cost + ob_cost + v_cost
                #print(v,y,final_cost)
                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    costs = [to_goal_cost,ob_cost,v_cost]
                    min_u = [v, y]

        return min_u,min_cost,costs


    def calc_obstacle_cost(self,traj, ob):
        # calc obstacle cost inf: collistion, 0:free

        minr = float("Inf")
        for i in range(10):
            oz = (2.0*i/10-1.0)*(np.pi/2.0)
            ox = ob[i]*np.cos(oz)
            oy = ob[i]*np.sin(oz)

            for j in range(len(traj[:,1])):
                dx = traj[j, 0] - ox
                dy = traj[j, 1] - oy
                r = np.sqrt(dx**2 + dy**2)
                if r <= self.robot_radius*1.1:
                    return float("Inf")  # collisiton

                if minr >= r:
                    minr = r

        return 1.0 / minr


    def calc_to_goal_cost(self,traj, goal):
        # calc to goal cost. It is 2D norm.

        dy = goal[0] - traj[-1, 0]
        dx = goal[1] - traj[-1, 1]
        goal_dis = math.sqrt(dx**2 + dy**2)
        cost = self.to_goal_cost_gain * goal_dis

        return cost

    def calc_vel_cost(self,v,y):
        return self.vel_cost_gain * v + self.yaw_cost_gain * abs(y)

    def dwa_control(self,u, state):
        #print(state)
            # Dynamic Window control
        lidar = np.zeros(10)
        for i in range(10):
            lidar[i] = state[i]
        #print(lidar)
        goal = np.array([state[10]*state[12], state[10]*state[11]])
        #print(goal)
        dw = self.calc_dynamic_window(u)
        #print(dw)
        u,min_cost,costs = self.calc_final_input(u, dw, goal,lidar)
        #print(min_cost,costs)
        return u

def main():

    for episode in range(100):
        state = env.reset()
        done = False
        ep_r = 0

        action = np.array([0.0, 0.0])
        expert = Expert()

        for t in range(1000):
            env.render()

            action_ = expert.dwa_control(action, state)
            #print(action_)
            state_ ,reward, done, info = env.step(action_)
            if t == 1000-1:
                done = True
            ep_r += reward
            action = action_
            state = state_
            #print(reward)
            if done:
                print("Episode %d finished after %d timesteps, reward: %f "%(episode+1,t+1,ep_r))
                break

if __name__ == '__main__':
    main()
