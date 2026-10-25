"""
Gymnasium Environments and Procedural Worlds for Quadruped Sim-to-Real RL.

Author: Youseef Abdel Tawwab
"""

from sim2real_rl.envs.terrain import (
    TerrainType,
    TerrainConfig,
    ProceduralTerrain,
)
from sim2real_rl.envs.rewards import (
    RewardConfig,
    RewardManager,
)
from sim2real_rl.envs.physics_engine import (
    PhysicsConfig,
    QuadrupedPhysicsEngine,
)
from sim2real_rl.envs.quadruped_env import (
    QuadrupedEnv,
)

# Optional Gymnasium registration
try:
    import gymnasium as gym
    if "QuadrupedLocomotion-v0" not in gym.envs.registry:
        gym.register(
            id="QuadrupedLocomotion-v0",
            entry_point="sim2real_rl.envs.quadruped_env:QuadrupedEnv",
            max_episode_steps=1000,
        )
except Exception:
    pass

__all__ = [
    "TerrainType",
    "TerrainConfig",
    "ProceduralTerrain",
    "RewardConfig",
    "RewardManager",
    "PhysicsConfig",
    "QuadrupedPhysicsEngine",
    "QuadrupedEnv",
]
