phynix_gym
==========

Environments to train reinforcement learning agents.

Currently contains one environment, `Minimize1DSimple`, minimizing negative log-likelihood fit to a gaussian sample.
The environment  takes a couple of configuration arguments and should therefore be instantiated from the class.
It adheres to the `openai gym API <https://gym.openai.com/>`_ and can also be instantiated with `gym.make("minimize-1d-simple-v0").

Example:

.. code-block:: python

  from phynix_gym import Minimize1DSimple

  env = Minimize1DSimple()
  
  nll, mu, mu_grad, sigma, sigma_grad = env.reset()  # return state
  state, reward, done, info = env.step([0.1, 0.3])  # corresponds to [mu, sigma]
  env.render()  # shows a nll contour plot, the position, gradient and the minimum

