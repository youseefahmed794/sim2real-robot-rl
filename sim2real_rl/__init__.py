"""
Sim-to-Real Reinforcement Learning for Dynamic Quadruped Locomotion.
Author: Youseef Abdel Tawwab
"""

from sim2real_rl.envs import (
    QuadrupedEnv,
    ProceduralTerrain,
    TerrainConfig,
    TerrainType,
    RewardManager,
    RewardConfig,
    PhysicsConfig,
    QuadrupedPhysicsEngine,
)

__version__ = "0.1.0"

__all__ = [
    "QuadrupedEnv",
    "ProceduralTerrain",
    "TerrainConfig",
    "TerrainType",
    "RewardManager",
    "RewardConfig",
    "PhysicsConfig",
    "QuadrupedPhysicsEngine",
]
