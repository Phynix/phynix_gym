import time

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

sess = None


def create_sess():
    global sess

    if sess is None:
        sess = tf.get_default_session()
        if sess is None:
            sess = tf.InteractiveSession()
    return sess


def loss_true_param_l2(env: "Minimize1DSimple", done: bool):
    reward = 10 if done else -0.1 * env.sess.run(env.param_loss_l2)
    return reward


def loss_minimum(env: "Minimize1DSimple", done: bool):
    reward = 10 if done else -0.01
    return reward


def loss_nll(env: "Minimize1DSimple", done: bool):
    reward = 10 if done else -0.1 * env.sess.run(env.nll)
    return reward


def done_combined(env: "Minimize1DSimple"):
    nll_tol = env.nll_tolerance
    if callable(nll_tol):
        nll_tol = nll_tol(env)

    param_tol = env.param_l2_tolerance
    if callable(param_tol):
        param_tol = param_tol(env)

    grad_tol = env.sum_grad_tolerance
    if callable(grad_tol):
        grad_tol = grad_tol(env)

    nll_conv = env.sess.run(env.nll_l1) < nll_tol
    param_conv = env.sess.run(env.param_loss_l2) < param_tol
    grad_conv = env.sess.run(env.sum_grads) < grad_tol
    converged = nll_conv and param_conv and grad_conv
    return converged


class Minimize1DSimple(gym.Env):
    _reward_funcs = {'true_param_l2': loss_true_param_l2,
                     'true_minimum': loss_minimum,
                     'loss': loss_nll
                     }
    _done_funcs = {'combined': done_combined}

    def __init__(self, reward='true_minimum', step_callbacks=None, param_low=-2, param_high=2, max_steps=100,
                 done_func='combined',
                 nll_tolerance=0.01, param_l2_tolerance=0.0001, sum_grad_tolerance=0.001,
                 grad_clip_low=-3, grad_clip_high=3, sample_param_limit_factor=0.95,
                 n_sample=10000,
                 plot_contour=True, plot_grad=True, plot_position=True, gridsize_axis_contour=30,

                 ):

        if step_callbacks is None:
            step_callbacks = []
        if not isinstance(step_callbacks, (list, tuple)):
            step_callbacks = [step_callbacks]
        self.step_callbacks = step_callbacks

        self.nll_tolerance = nll_tolerance
        self.param_l2_tolerance = param_l2_tolerance
        self.sum_grad_tolerance = sum_grad_tolerance

        self.figure = None
        self._plotting_is_reset = False
        self.plot_contour = plot_contour
        self.plot_grad = plot_grad
        self.plot_position = plot_position
        self.gridsize_axis_contour = gridsize_axis_contour

        if isinstance(reward, str):
            reward = self._reward_funcs[reward]
        if not callable(reward):
            raise TypeError("reward has to be callable")
        self.calc_reward = reward

        if isinstance(done_func, str):
            done_func = self._done_funcs[done_func]
        if not callable(done_func):
            raise TypeError("done_func has to be callable")
        self.done_func = done_func

        self.n_tot_steps = 0
        self.n_steps = 0
        self.max_steps = max_steps

        self.sess = create_sess()
        self.n_sample = n_sample
        self.n_params = 2
        nll_norm = 80000 * self.n_sample / 3000  # nice normalization
        nll_low = -10
        nll_high = 10
        self.param_low = param_low
        self.param_high = param_high
        self.observation_space = gym.spaces.Box(low=np.array([nll_low] + [param_low, grad_clip_low] * self.n_params),
                                                high=np.array(
                                                        [nll_high] + [param_high, grad_clip_high] * self.n_params),
                                                dtype=np.float64)
        self.action_space = gym.spaces.Box(low=np.array([param_low] * self.n_params),
                                           high=np.array([param_high] * self.n_params,
                                                         dtype=np.float64))

        self.params = [tf.get_variable("var_" + str(i) + str(np.random.randint(low=0, high=1e12)),
                                       initializer=tf.random_uniform(shape=(),
                                                                     minval=self.param_low,
                                                                     maxval=self.param_high,
                                                                     dtype=tf.float64))
                       for i in range(self.n_params)]
        self.sample_params = [tf.get_variable("var_sample_" + str(i) + str(np.random.randint(low=0, high=1e12)),
                                              initializer=tf.random_uniform(shape=(),
                                                                            minval=self.param_low *
                                                                                   sample_param_limit_factor,
                                                                            maxval=self.param_high *
                                                                                   sample_param_limit_factor,
                                                                            dtype=tf.float64))
                              for i in range(self.n_params)]
        sigma = self.params[1]
        sigma_sample = self.sample_params[1]
        sigma = (sigma - self.param_low + 1e-8) / (self.param_high + 4e-8)
        sigma_sample = (sigma_sample - self.param_low + 1e-8) / (self.param_high + 4e-8)
        self.dist = tfp.distributions.Normal(loc=np.array(self.params)[0],
                                             scale=sigma)
        self.sample_dist = tfp.distributions.Normal(loc=np.array(self.sample_params)[0],
                                                    scale=sigma_sample)
        sample = self.sample_dist.sample(sample_shape=(self.n_sample,))
        self.sample = tf.get_variable(name="sample", trainable=False, initializer=sample, use_resource=True)
        self.nll = - tf.reduce_sum(self.dist.log_prob(self.sample)) / nll_norm
        self.true_nll = - tf.reduce_sum(self.sample_dist.log_prob(self.sample)) / nll_norm
        self.nll_l1 = tf.abs(self.true_nll - self.nll)
        self.gradients = tf.gradients(self.nll, self.params)
        params_grads = []
        for param, grad in zip(self.params, self.gradients):
            params_grads.append(param)
            params_grads.append(grad)
        clipped_nll = tf.clip_by_value(self.nll, nll_low, nll_high)
        self.observations = [clipped_nll] + params_grads
        self.param_loss_l2 = tf.losses.mean_squared_error(labels=self.sample_params,
                                                          predictions=self.params)
        self.sum_grads = tf.reduce_sum(tf.abs(self.gradients))

    def step(self, action):
        self.n_tot_steps += 1
        self.n_steps += 1

        for param, a in zip(self.params, action):
            param.load(value=a, session=self.sess)

        obs = self._get_obs()
        # if np.random.random() < 0.0001:
        #     print("loss:", param_loss, " params", action, "n tot steps", self.n_tot_steps)

        # running_norm = max(min((self.n_tot_steps / 100000), 1.), 0.01)
        # done = loss < (0.01 / running_norm)

        done = self.done_func(self)
        reward = self.calc_reward(env=self, done=done)
        for callback in self.step_callbacks:
            callback(self, done=done)
        # done = obs[0] < 0.000001
        # reward = 10 if done else -0.1 * loss
        # print("reward", reward)
        # reward = 10 if done else -0.01
        # reward += penalty
        # if done:
        # print("DDDDDDOOOOOONNNNNNNNEEEEE!!!!!!!!!, n steps:", self.n_steps, "param_loss", param_loss,
        #       "observables:", self._get_obs(), "true_params", self.sess.run(self.sample_params),
        #       "sum gradients", sum_grads)
        if self.n_steps >= self.max_steps:
            done = True
        # if self.n_tot_steps % 10000 <= 300:
        #     self.render()
        #     time.sleep(0.025)
        return obs, reward, done, {}

    def _get_obs(self):
        return self.sess.run(self.observations)

    def _prepare_plotting(self):
        if self._plotting_is_reset:
            return
        figure = plt.figure("NLL plot")
        plt.title("NLL vs params plot")
        plt.xlabel("mu")
        plt.ylabel("sigma (transformed)")
        self.figure = figure
        self.figure_ax = self.figure.add_subplot(111)
        self.figure_line, = self.figure_ax.plot([self.param_low, self.param_high],
                                                [self.param_low, self.param_high],
                                                "rx")
        if self.plot_contour:
            self._plot_contour()
        self._plotting_is_reset = True

    def _plot_contour(self):
        num = self.gridsize_axis_contour
        old_params = self.sess.run(self.params)
        x_values = np.linspace(self.param_low * 0.99,
                               self.param_high * 0.99,
                               num=num)
        y_values = x_values
        nll_values = np.zeros(shape=(num, num))

        for i, x in enumerate(x_values):
            self.params[0].load(session=self.sess, value=x)
            for j, y in enumerate(y_values):
                self.params[1].load(session=self.sess, value=y)
                nll = self.sess.run(self.nll)
                nll_values[j, i] = min(nll, 0.3)
                # nll_values[i, j] = f(x, y)
        min_index = np.unravel_index(np.argmin(nll_values), nll_values.shape)
        print("minimum grid search", np.min(nll_values), 'at',
              x_values[min_index[0]], y_values[min_index[1]])
        true_xy = self.sess.run(self.sample_params)

        nll_true = self.sess.run(self.true_nll)
        print("nll true val", nll_true, 'at', true_xy)
        plt.contourf(x_values, y_values, nll_values, cmap='RdGy')
        plt.plot(true_xy[0], true_xy[1], 'rx')
        plt.colorbar()

        # set old values again

        for param, old_val in zip(self.params, old_params):
            param.load(value=old_val, session=self.sess)

        return

    def reset(self):
        if self.figure is not None:
            plt.clf()

        self._plotting_is_reset = False

        for param in self.sample_params + self.params:
            self.sess.run(param.initializer)
        self.sess.run(self.sample.initializer)

        # print("Environment reset, true params:", self.sess.run(self.sample_params))
        self.n_steps = 0

        return self._get_obs()

    def render(self, mode='human'):
        self._prepare_plotting()
        if self.n_params != 2:
            raise ValueError("Can only plot if there are 2 params.")

        if self.plot_position:
            param1, param2 = self.sess.run(self.params)
            true_param1, true_param2 = self.sess.run(self.sample_params)
            if self.plot_grad:
                grad = self.sess.run(self.gradients)
                grad_scale = np.array([0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 1.]) * 10
                param1_grad_points = grad_scale * grad[0] + param1
                param2_grad_points = grad_scale * grad[1] + param2
            else:
                param1_grad_points = []
                param2_grad_points = []

            self.figure_line.set_xdata([param1, true_param1] + list(param1_grad_points))
            self.figure_line.set_ydata([param2, true_param2] + list(param2_grad_points))

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.show(block=False)


if __name__ == '__main__':
    env = Minimize1DSimple()
    nll, param1, grad1, param2, grad2 = env.reset()
    print("action space size", env.action_space.shape)
    lower = env.action_space.low
    upper = env.action_space.high
    # env._plot_contour()

    scale = 3.
    for i in range(1000):
        step_output = env.step([param1 - scale * grad1,
                                param2 - scale * grad2])
        (nll, param1, grad1, param2, grad2), r, d, _ = step_output
        print("step output", step_output)
        env.render()
        if d:
            reset_output = env.reset()
            nll, param1, grad1, param2, grad2 = reset_output
            print("reset output", reset_output)

            # env._plot_contour()
