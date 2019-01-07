# phynix_gym
Environments to train reinforcement learning agents.

Currently contains one environment, `Minimize1DSimple`, minimizing negative log-likelihood fit to a gaussian sample.
The environment  takes a couple of configuration arguments and should therefore be instantiated from the class.
It adheres to the `openai gym API <https://gym.openai.com/>`_ and can also be instantiated with `gym.make("minimize-1d-simple-v0").

example:
```python
from phynix_gym import Minimize1DSimple

env = Minimize1DSimple()
env.reset()
```
