"""
Reinforcement Learning Algorithms and Policy Architectures.

Exposes:
- ActorCritic: Multi-layer perceptron with Gaussian policy & state-value heads
- RolloutBuffer: On-policy buffer with GAE advantage computation
- PPO: Proximal Policy Optimization algorithm with clipped objective
- PPOConfig: Configuration dataclass for PPO hyperparameters

Author: Youseef Abdel Tawwab
"""

from .network import ActorCritic, NumpyMLP
from .buffer import RolloutBuffer
from .ppo import PPO, PPOConfig

__all__ = [
    "ActorCritic",
    "NumpyMLP",
    "RolloutBuffer",
    "PPO",
    "PPOConfig",
]
