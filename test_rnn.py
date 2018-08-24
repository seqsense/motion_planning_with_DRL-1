import tensorflow as tf
import numpy as np
import gym
import myenv
from train_rnn import ACNet
GAME = 'myenv-v0'
MAX_EP_STEP = 1000
MAX_EP = 10
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
GLOBAL_NET_SCOPE = 'Global_Net'
NN_MODEL = './models/nn_model_ep_533300.ckpt'
env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]


def main():
    #with tf.Session() as sess:
    sess = tf.Session()
    with tf.device("/cpu:0"):
        global_ac = ACNet(sess,GLOBAL_NET_SCOPE)
        test_ac = ACNet(sess,"W_0",global_ac)
        saver = tf.train.Saver()
        saver.restore(sess, NN_MODEL)
    
        for ep in range(MAX_EP):
            s = env.reset()
            ep_r = 0
            rnn_state = sess.run(test_ac.init_state)
            for t in range(MAX_EP_STEP):
                env.render()
                a, rnn_state_ = test_ac.choose_action(s,rnn_state)

                s_, r, done, info = env.step(a)
                if t ==  MAX_EP_STEP-1:
                    done = True
                ep_r += r
                s = s_
                rnn_state = rnn_state_

                if done:
                    break
            print(ep, ep_r, done,t)


if __name__ == '__main__':
    main()
