import gym
from gym import error, spaces, utils
from gym.utils import seeding

import tensorflow as tf



class Minimize1DNP(gym.Env):
    def __init__(self):
        sess = tf.get_default_session()
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    Minimize1DNP()
