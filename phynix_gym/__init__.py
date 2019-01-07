from gym.envs.registration import register

__version__ = "0.0.3"
__author__ = "Jonas Eschle 'Mayou36'"
__email__ = "jonas.eschle@cern.ch"

register(
        id='minimize-1d-simple-v0',
        entry_point='phynix_gym.envs:Minimize1DSimple',
        )
from .envs import Minimize1DSimple
