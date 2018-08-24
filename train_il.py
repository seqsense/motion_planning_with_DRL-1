import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import myenv
import os
import shutil
from datetime import datetime
from dwa import Expert
import matplotlib.pyplot as plt

GAME = 'myenv-v0'
OUTPUT_GRAPH = True
LOG_DIR = './il/log'
MODEL_DIR = './il/models'
SUMMARY_DIR = './il/results'
MODEL_SAVE_INTERVAL = 100
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEP = 1000
MAX_GLOBAL_EP = 1000000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER =1 
GAMMA = 0.99
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
#NN_MODEL = './models/nn_model_ep_10000.ckpt'
NN_MODEL = None

env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]


def build_summaries():
    value = tf.Variable(0.)
    tf.summary.scalar("Value", value)
    #actor_loss = tf.Variable(0.)
    #tf.summary.scalar("Actor_loss", actor_loss)
    #critic_loss = tf.Variable(0.)
    #tf.summary.scalar("Critic_loss", critic_loss)
    eps_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_reward", eps_reward)

    #summary_vars = [actor_loss,critic_lhoss, eps_total_reward]
    summary_vars = [value, eps_reward]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

class ACNet(object):
    def __init__(self,sess, scope, globalAC=None):
        self.sess = sess
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                self.ex_a = tf.placeholder(tf.float32, [None, N_A], 'EX_A')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                self.td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(self.td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4

                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1),[0,1]), A_BOUND[0], A_BOUND[1])
                
                with tf.name_scope('a_loss'):
                    self.a_loss = tf.reduce_mean(tf.subtract(self.ex_a,self.A))

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = tf.train.AdamOptimizer(LR_A).apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = tf.train.AdamOptimizer(LR_C).apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('critic'):   # only critic controls the rnn update
            cell_size = 16
            s = tf.expand_dims(self.s, axis=1,
                               name='timely_input')  # [time_step, feature] => [time_step, batch, feature]
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(cell_size)
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell, inputs=s, initial_state=self.init_state, time_major=True)
            cell_out = tf.reshape(outputs, [-1, cell_size], name='flatten_rnn_outputs')  # joined state representation
            l_c1 = tf.layers.dense(cell_out, 512, tf.nn.relu, kernel_initializer=w_init, name='lc1')
            l_c2 = tf.layers.dense(l_c1, 512, tf.nn.relu, kernel_initializer=w_init, name='lc2')
            l_c3 = tf.layers.dense(l_c2, 512, tf.nn.relu, kernel_initializer=w_init, name='lc3')
            v = tf.layers.dense(l_c3, 1, kernel_initializer=w_init, name='v')  # state value

        with tf.variable_scope('actor'):  # state representation is based on critic
            l_a1 = tf.layers.dense(cell_out, 512, tf.nn.relu, kernel_initializer=w_init, name='la1')
            l_a2 = tf.layers.dense(l_a1, 512, tf.nn.relu, kernel_initializer=w_init, name='la2')
            l_a3 = tf.layers.dense(l_a2, 512, tf.nn.relu, kernel_initializer=w_init, name='la3')
            mu = tf.layers.dense(l_a3, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a3, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, cell_state):  # run by a local
        s = s[np.newaxis, :]
        a, cell_state = self.sess.run([self.A, self.final_state], {self.s: s, self.init_state: cell_state})
        return a, cell_state

    def get_loss(self,feed_dict):
        c_loss = self.sess.run(self.c_loss,feed_dict)
        a_loss = self.sess.run(self.a_loss,feed_dict)
        return c_loss,a_loss


class Worker(object):
    def __init__(self,sess, name, globalAC):
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.sess = sess
        self.AC = ACNet(sess,name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        buffer_ex_a = []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            rnn_state = self.sess.run(self.AC.init_state)    # zero rnn state at beginning
            keep_state = rnn_state       # keep rnn state for updating global net
            a = np.array([0.0,0.0])
            expert = Expert()
            for ep_t in range(MAX_EP_STEP):
                if self.name == 'W_0':
                    self.env.render()
                expert_a = expert.dwa_control(a,s)
                a_, rnn_state_ = self.AC.choose_action(s, rnn_state)  # get the action and next rnn state
                s_, r, done, info = self.env.step(a_)
                if ep_t == MAX_EP_STEP- 1:
                    r = -3.0
                    done = True

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a_)
                buffer_r.append((r+8)/8)    # normalize
                buffer_ex_a.append(expert_a)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :], self.AC.init_state: rnn_state_})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s = np.vstack(buffer_s)
                    buffer_a = np.vstack(buffer_a)
                    buffer_v_target = np.vstack(buffer_v_target)
                    buffer_ex_a = np.vstack(buffer_ex_a)

                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        self.AC.init_state: keep_state,
                        self.AC.ex_a: buffer_ex_a,
                    }

                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    buffer_ex_a = []
                    self.AC.pull_global()
                    keep_state = rnn_state_  # replace the keep_state as the new initial rnn state_

                s = s_
                rnn_state = rnn_state_  # renew rnn state
                total_step += 1
                a = a_

                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %f" % GLOBAL_RUNNING_R[-1],
                        "| ep_r: %f" % ep_r,
                        "| ep_t: %f" % ep_t,
                          )
                    GLOBAL_EP += 1
                    break
            #c_loss,a_loss = self.AC.get_loss(feed_dict)
            summary_str = self.sess.run(summary_ops, feed_dict={
                summary_vars[0]: v_s_,
                summary_vars[1]: GLOBAL_RUNNING_R[-1]
                })
            writer.add_summary(summary_str,GLOBAL_EP)
            writer.flush()

            if GLOBAL_EP % MODEL_SAVE_INTERVAL == 0:
                save_path = saver.save(self.sess,MODEL_DIR + "/nn_model_ep_" + str(GLOBAL_EP) + ".ckpt")

if __name__ == "__main__":
    sess = tf.Session()

    with tf.device("/cpu:0"):
        GLOBAL_AC = ACNet(sess,GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(sess,i_name, GLOBAL_AC))
            
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    logdir = "{}/run-{}".format(SUMMARY_DIR,now)
    summary_ops, summary_vars = build_summaries()
    COORD = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
    saver = tf.train.Saver()

    nn_model = NN_MODEL
    if nn_model is not None:  # nn_model is the path to file
        saver.restore(sess, nn_model)
        print("Model restored.")

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, sess.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
