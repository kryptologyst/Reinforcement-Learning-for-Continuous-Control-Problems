"""RL Continuous Control Package."""

__version__ = "1.0.0"
__author__ = "RL Research Team"
__email__ = "research@example.com"

from .ddpg import DDPGAgent, Actor, Critic, ReplayBuffer, OUNoise
from .sac import SACAgent, GaussianPolicy, QNetwork

__all__ = [
    "DDPGAgent",
    "Actor", 
    "Critic",
    "ReplayBuffer",
    "OUNoise",
    "SACAgent",
    "GaussianPolicy",
    "QNetwork"
]
