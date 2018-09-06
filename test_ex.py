import tensorflow as tf
import numpy as np
import gym
import myenv
from train_ex import PPONet
GAME = 'myenv-v2'
MAX_EP_STEP = 1000
MAX_EP = 10
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
NN_MODEL = './models/ex/ppo_model_ep_14000.ckpt'
env = gym.make(GAME)

NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.shape[0]
A_BOUNDS = [env.action_space.low, env.action_space.high]
NUM_HIDDEN = [512,512,512]
EPSILON = 0.2

def main():
    #with tf.Session() as sess:
    sess = tf.Session()
    with tf.device("/cpu:0"):
        brain = PPONet(sess)
        saver = tf.train.Saver()
        saver.restore(sess, NN_MODEL)
    
        for ep in range(MAX_EP):
            s = env.reset().reshape(-1)
            ep_r = 0
            for t in range(MAX_EP_STEP):
                env.render()

                s = np.array([s])
                a = brain.predict_a(s).reshape(-1)

                s_, r, done, info = env.step(a)
                if t ==  MAX_EP_STEP-1:
                    done = True
                ep_r += r
                s = s_.reshape(-1)
                if done:
                    break
            print(ep, ep_r, done,t)


if __name__ == '__main__':
    main()
