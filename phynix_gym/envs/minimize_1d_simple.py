import collections
import copy
from typing import Union, Callable, Iterable, List, Tuple

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd

sess = None


def create_sess():
    global sess

    if sess is None:
        sess = tf.get_default_session()
        if sess is None:
            sess = tf.InteractiveSession()
    return sess


def loss_true_param_l2(env: "Minimize1DSimple", done: bool):
    # reward = 10 if done else -0.1 * env.sess.run(env.param_loss_l2)
    param_loss_l2 = env.sess.run(env.param_loss_l2)
    alpha = 0.5
    if done:
        reward = 50
    else:
        reward = -1 * (np.sqrt(param_loss_l2) * (1 - alpha) + alpha * param_loss_l2)
        reward -= 0.2  # time punishment
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
                     'true_minimum_binary': loss_minimum,
                     'loss': loss_nll
                     }
    _done_funcs = {'combined': done_combined}

    def __init__(self, reward: Union[str, Callable] = 'true_minimum_binary', reward_scale: float = 1.,
                 step_callbacks: Iterable[Callable] = None, param_low: float = -2, param_high: float = 2,
                 max_steps: int = 100, n_old_obs: int = 0,
                 pad_missing_obs: Union[bool, float, Iterable[float], str] = False,
                 done_func: Union[str, Callable] = 'combined',
                 nll_tolerance: float = 0.1, param_l2_tolerance: float = 0.05, sum_grad_tolerance: float = 0.05,
                 grad_clip_low: int = -3, grad_clip_high: int = 3, sample_param_limit_factor: float = 0.95,
                 n_sample: int = 10000,
                 plot_contour: bool = True, plot_grad: bool = True, plot_position: bool = True,
                 gridsize_axis_contour: int = 30,

                 ):
        """Minimize a gaussian maximum likelihood (NLL) fit to a sample from a gaussian with two free parameters.

        The sample is drawn and a negative log-likelihood is created. The state is a list of float consisting of:
        ```
        [nll, mu, grad(nll, mu), sigma, grad(nll, sigma)]
        ```
        and, if n_old_obs is specified, it also contains the `n_old_obs` previous states
        ```
        [nll, mu, grad_mu, sigma, grad_sigma, nll_prev, mu_prev, grad_mu_prev, sigma_prev, grad_sigma_prev,...]
        ```
        and the action takes a list of values between `param_low` and `param_high`: [mu, sigma] (*sigma is actually
        scaled, so it's not a 1-to-1 translation).

        The environment is solved, if the parameters are "close enough" (see also `done_func`) to the true
        parameters the sample was drawn from.

        Algorithms sometimes work better with symmetric limits, this can be done here.

        `render` plots a nll contour, a goal and a current position.

        To have a callback in each step, use the `callback` argument to provide (a list of) callback functions.



        Args:
            reward (str, callable): The reward function can be a str or a callable. For a string,
            the following are implemented (for any is valid: finding the minimum yields 10):

                - 'true_param_l2': reward is the negative euclidean distance of the current set parameters to the
                    true parameters
                - 'true_minimum_binary': return a binary reward: -0.01 if not at the minimum, 10 if at the minimum
                - 'loss': return the negative, current loss. The smaller the loss, the better.
            reward_scale (float): a scale applied to any reward. default to 1.
            step_callbacks (Iterable[Callable]): An iterable containing callback functions that get executed at every
                step. The function gets two arguments, the environment itself and a boolean "done" flag.
            param_low (int): The lower limit of the parameters (sigma is moved/scaled into the positive area)
            param_high (int): The upper limits of the parameters (best: symmetric)
            max_steps (int): Maximum number of steps until the environment gets reset
            n_old_obs (int): The observation should conists of the current observations and `n_old_obs` old observations
                concatenated to the right
            pad_missing_obs (Union[bool, float, list[float, ...], str]): dynamic shape vs padding. How to deal with old
            observations.
                If False, the returned observations
                shape will change. At the beginning, it will be just the observations and grow by appending the previous
                observations up to `n_old_obs`. Otherwise, the observation shape will always be constant and padded with
                `pad_pad_missing_obs`. This is either a single float or a list with len(observations).
                To pad the values with the first observation, use the string "initial".
            done_func (Union[str, Callable]): The criteria when the environment is done -> definition of the minimum.
                A str can be given: "combined". This is also the default function with the criterion that **all three**
                (nll difference to true nll, param l2 difference to true param and sum of the gradients)
                have to be below certain thresholds (nll_tolerance, param_l2_tolerance and sum_grad_torelance).
                A custom function can be used and should return a boolean "done".
            nll_tolerance (float): Used for `abs(nll-nll_true) < nll_tolerance` as one of the convergences criterion
                in the default `done_func`
            param_l2_tolerance (float): Same as above: l2_norm(param - param_true) < param_l2_tolerance
            sum_grad_tolerance (float): Same as above: sum(gradients) < sum_grad_tolerance
            grad_clip_low (int): Lower limit to clip the gradient
            grad_clip_high (int): Upper limit to clip the gradient
            sample_param_limit_factor (float): A factor to limit the parameter sampling range to prevent to
                sample from the limit (typically 0.9..0.95 is good)
            n_sample (int): How many samples to draw
            plot_contour (bool): If True, plot the nll contour when using `render`
            plot_grad (bool): If True, plot the gradient when using `render`. Several crosses are used with different
                spacing: the gradient points into the direction of denser crosses (if contour is used: red means low)
            plot_position (bool): If True, plot the current position when using `render`
            gridsize_axis_contour (int): How many points to draw per axis when plotting the contour. This will result
                in `gridsize_axis_contour^2` nll evaluations.
        """
        dist_normal = tfd.Normal
        mix = tf.constant(0.4, dtype=tf.float64)

        def dist_2gauss(param1, param2):
            return tfd.Mixture(cat=tfd.Categorical(probs=[mix, 1. - mix]),
                               components=[
                                   tfd.Normal(loc=0.1, scale=param2),
                                   tfd.Normal(loc=param1, scale=3),
                                   # tfd.Exponential(rate=tf.constant(1., dtype=tf.float64)),
                                   ])

        distribution = dist_normal
        # distribution = dist_2gauss

        self.n_old_obs = n_old_obs
        self.pad_missing_obs = pad_missing_obs
        self.old_observations = collections.deque([])

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
        self._plot_show_done = False
        self.plot_contour = plot_contour
        self.plot_grad = plot_grad
        self.plot_position = plot_position
        self.gridsize_axis_contour = gridsize_axis_contour

        self._count_success = 0
        self._count_failure = 0

        if isinstance(reward, str):
            reward = self._reward_funcs[reward]
        if not callable(reward):
            raise TypeError("reward has to be callable")
        self.calc_reward = reward
        self.reward_scale = reward_scale

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

        grad_scale_low = 1e-7
        grad_scale_high = 10

        lower_observation_limits = [nll_low] * 2 + [param_low, grad_clip_low] * self.n_params + [grad_scale_low]
        upper_observation_limits = [nll_high] * 2 + [param_high, grad_clip_high] * self.n_params + [grad_scale_high]
        self.observation_space = gym.spaces.Box(low=np.array(lower_observation_limits * (1 + n_old_obs)),
                                                high=np.array(upper_observation_limits * (1 + n_old_obs)),
                                                dtype=np.float64)
        self.action_space = gym.spaces.Box(low=np.array([param_low] * self.n_params),
                                           high=np.array([param_high] * self.n_params,
                                                         dtype=np.float64))

        param_names = ['mu', 'sigma']
        self.params = [tf.get_variable("var_{}".format(name) + str(i) + str(np.random.randint(low=0, high=1e12)),
                                       initializer=tf.random_uniform(shape=(),
                                                                     minval=self.param_low,
                                                                     maxval=self.param_high,
                                                                     dtype=tf.float64))
                       for i, name in zip(range(self.n_params), param_names)]
        self.sample_params = [
            tf.get_variable("var_sampl_{}_".format(name) + str(i) + str(np.random.randint(low=0, high=1e12)),
                            initializer=tf.random_uniform(shape=(),
                                                          minval=self.param_low *
                                                                 sample_param_limit_factor,
                                                          maxval=self.param_high *
                                                                 sample_param_limit_factor,
                                                          dtype=tf.float64))
            for i, name in zip(range(self.n_params), param_names)]
        sigma = self.params[1]
        sigma_sample = self.sample_params[1]
        sigma = (sigma - self.param_low + 1e-8) / (self.param_high + 4e-8)
        sigma_sample = (sigma_sample - self.param_low + 1e-8) / (self.param_high + 4e-8)
        # self.dist = distribution(np.array(self.params)[0], sigma)
        self.dist = distribution(self.params[0], sigma)
        self.sample_dist = distribution(self.sample_params[0], sigma_sample)
        sample = self.sample_dist.sample(sample_shape=(self.n_sample,))
        self.sample = tf.get_variable(name="sample" + str(np.random.randint(low=0, high=1e12)), trainable=False,
                                      initializer=sample, use_resource=True)
        self.nll = - tf.reduce_sum(self.dist.log_prob(self.sample)) / nll_norm
        self.true_nll = - tf.reduce_sum(self.sample_dist.log_prob(self.sample)) / nll_norm
        self.nll_l1 = tf.abs(self.true_nll - self.nll)
        self.previous_nll = tf.get_variable(name="previous_nll" + str(np.random.randint(low=0, high=1e12)),
                                            trainable=False, initializer=self.nll,
                                            use_resource=False)
        self.delta_nll = self.nll - self.previous_nll
        # self.gradients = [tf.sqrt(tf.abs(grad)) * tf.sign(grad) for grad in tf.gradients(self.nll, self.params)]
        gradients = tf.gradients(self.nll, self.params)
        self.sum_grads = tf.reduce_sum(tf.abs(gradients))
        gradients_norm = tf.norm(gradients)
        gradients = [grad / gradients_norm for grad in gradients]
        self.gradients = gradients
        self.gradients_norm = tf.sqrt(tf.log1p(gradients_norm))

        params_grads = []
        for param, grad in zip(self.params, self.gradients):
            params_grads.append(param)
            params_grads.append(- grad)
        clipped_nll = tf.clip_by_value(self.nll, nll_low, nll_high)
        clipped_delta_nll = tf.clip_by_value(self.delta_nll, nll_low, nll_high)

        self.observations = [clipped_nll, clipped_delta_nll] + params_grads + [self.gradients_norm]
        self.param_loss_l2 = tf.losses.mean_squared_error(labels=self.sample_params,
                                                          predictions=self.params)
        self.param_loss_l2 = tf.sqrt(self.param_loss_l2)

    def step(self, action: List[float]) -> Tuple[List[float], float, bool, dict]:
        """Set new parameters [mu, sigma]

        Args:
            action (list[float, float]): The new values of mu, sigma

        Returns:
            observations, reward, done, info: For further discription, see the docs of the environment (`__init__`).
        """
        self.n_tot_steps += 1
        self.n_steps += 1
        info = {}

        self.previous_nll.load(session=self.sess, value=self.sess.run(self.nll))

        for param, a in zip(self.params, action):
            param.load(value=a, session=self.sess)

        obs = self._get_obs()

        if np.random.random() < 0.0001:
            print("values: ", action)

        done = self.done_func(self)
        reward = self.calc_reward(env=self, done=done) * self.reward_scale
        if done:
            info['n_steps'] = self.n_steps
            self._count_success += 1
            if self._count_success % 50:
                print("n_steps needed:", info)
        for callback in self.step_callbacks:
            callback(self, done=done)

        if self.n_steps >= self.max_steps:
            done = True
            self._count_failure += 1
            if self._count_failure % 50:
                print("failed to find minimum")

        return obs, reward, done, info

    def _get_obs(self):
        current_obs = self.sess.run(self.observations)
        if self.n_old_obs:

            observations = copy.deepcopy(self.old_observations)
            observations.extendleft(reversed(current_obs))
            if len(self.old_observations) < self.n_old_obs * len(self.observations):
                self.old_observations.extendleft(current_obs)
            else:
                self.old_observations.rotate(len(self.observations))  # move the right most (old) to the beginning
                for i, obs in enumerate(current_obs):  # overwrite them here
                    self.old_observations[i] = obs
        else:
            observations = current_obs
        return observations

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
        if not self._plot_show_done:
            plt.show(block=False)
            self._plot_show_done = True

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

    def reset(self):
        """Reset the environment: draw a new sample with new parameters and return the current state.

        Returns:
            state: for more information, read the class docs (in `__init__`)
        """
        if self.figure is not None:
            plt.clf()

        self._plotting_is_reset = False

        # self._count_failure = 0
        # self._count_success = 0
        for param in self.sample_params + self.params:
            self.sess.run(param.initializer)
        self.sess.run(self.sample.initializer)
        self.sess.run(self.previous_nll.initializer)

        # for memory of old observations
        if self.pad_missing_obs is not False:
            if self.pad_missing_obs == "initial":
                temp_n_old_obs = self.n_old_obs
                self.n_old_obs = 0
                self.pad_missing_obs = self._get_obs()
                self.n_old_obs = temp_n_old_obs
            elif isinstance(self.pad_missing_obs, str):
                raise ValueError("Invalid str argument: ", self.pad_missing_obs)

            try:
                if len(self.pad_missing_obs) == len(self.observations):
                    padding = self.pad_missing_obs
                elif len(self.pad_missing_obs) == 1:
                    padding = list(self.pad_missing_obs) * len(self.observations)
                else:
                    raise ValueError("pad_missing_obs has to have length 1 or len(observations)")
            except TypeError:
                padding = [self.pad_missing_obs] * len(self.observations)
            old_obs = padding * self.n_old_obs
        else:
            old_obs = []

        self.old_observations = collections.deque(old_obs)

        # print("Environment reset, true params:", self.sess.run(self.sample_params))
        self.n_steps = 0

        return self._get_obs()

    def render(self, mode='human'):
        """Plot the two params a) target position, b) current position, c) gradient and a nll contour plot.

        What is plotted can be changed by changing the appropriate attributes (or in the `__init__`).

        Args:
            mode (): Only 'human' available.
        """
        if mode != 'human':
            raise NotImplementedError("anything else then 'human' is not implemented.")

        self._prepare_plotting()
        if self.n_params != 2:
            raise ValueError("Can only plot if there are 2 params.")

        if self.plot_position:
            param1, param2 = self.sess.run(self.params)
            true_param1, true_param2 = self.sess.run(self.sample_params)
            if self.plot_grad:
                grad, grad_norm = self.sess.run([self.gradients, self.gradients_norm])

                grad_scale = np.array([0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 1.]) * grad_norm * 10
                param1_grad_points = grad_scale * grad[0] + param1
                param2_grad_points = grad_scale * grad[1] + param2
            else:
                param1_grad_points = []
                param2_grad_points = []

            self.figure_line.set_xdata([param1, true_param1] + list(param1_grad_points))
            self.figure_line.set_ydata([param2, true_param2] + list(param2_grad_points))

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


if __name__ == '__main__':
    print("Running test environment with render and a 'dummy' vanilla gradient descent.")
    env = Minimize1DSimple()
    nll, nll_diff, param1, grad1, param2, grad2, grad_norm, *_ = env.reset()
    print("action space size", env.action_space.shape)
    lower = env.action_space.low
    upper = env.action_space.high
    # env._plot_contour()

    scale = 0.3
    for i in range(1000):
        env.render()
        step_output = env.step([param1 + scale * grad1 * grad_norm,
                                param2 + scale * grad2 * grad_norm])
        (nll, nll_diff, param1, grad1, param2, grad2, grad_norm, *_d), r, d, _ = step_output
        print("step output", step_output)
        print("l2 params", np.sqrt(env.sess.run(env.param_loss_l2)))
        if d:
            reset_output = env.reset()
            nll, param1, grad1, param2, grad2, *_ = reset_output
            print("reset output", reset_output)

            # env._plot_contour()
