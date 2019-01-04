import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class Minimize2DSimple(gym.Env):
    def __init__(self):
        sess = tf.get_default_session()
        if sess is None:
            sess = tf.Session()
        self.sess = sess
        self.n_sample = 10000
        self.n_params = 2
        nll_low = -100000
        nll_high = 100000
        param_low, param_high = -1, 1
        grad_low, grad_high = -1, 1
        self.param_low = param_low
        self.param_high = param_high
        self.observation_space = gym.spaces.Box(low=np.array([nll_low] + [param_low, grad_low] * self.n_params),
                                                high=np.array([nll_high] + [param_high, grad_high] * self.n_params),
                                                dtype=np.float64)
        self.action_space = gym.spaces.Box(low=np.array([param_low] * self.n_params),
                                           high=np.array([param_high] * self.n_params,
                                                         dtype=np.float64))

        self.params = [tf.get_variable("var_" + str(i),
                                       initializer=tf.random_uniform(shape=(),
                                                                     minval=self.param_low,
                                                                     maxval=self.param_high,
                                                                     dtype=tf.float64))
                       for i in range(self.n_params)]
        self.sample_params = [tf.get_variable("var_sample_" + str(i),
                                              initializer=tf.random_uniform(shape=(),
                                                                            minval=self.param_low,
                                                                            maxval=self.param_high,
                                                                            dtype=tf.float64))
                              for i in range(self.n_params)]
        self.dist = tfp.distributions.Normal(loc=np.array(self.params)[0],
                                             scale=np.array(self.params)[1])
        self.sample_dist = tfp.distributions.Normal(loc=np.array(self.sample_params)[0],
                                                    scale=np.array(self.sample_params)[1])
        self.sample = self.sample_dist.sample(sample_shape=(self.n_sample,))
        self.nll = tf.reduce_sum(self.dist.log_prob(self.sample))
        self.gradients = tf.gradients(self.nll, self.params)

    def step(self, action):
        for a in action:
            self.param.load(value=a, session=self.sess)

    def reset(self):

        for param in self.sample_params + self.params:
            self.sess.run(param.initializer)

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    env = Minimize2DSimple()
    env.reset()
