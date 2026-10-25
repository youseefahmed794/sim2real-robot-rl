"""
Unit tests for Native MuJoCo Quadruped Environment.

Author: Youseef Abdel Tawwab
"""

import numpy as np
import pytest

from sim2real_rl.envs.mujoco_env import MuJoCoQuadrupedEnv


def test_mujoco_env_initialization():
    """Verify MuJoCo model loading, dimensions, and initial state."""
    env = MuJoCoQuadrupedEnv()
    assert env.observation_dim == 48
    assert env.action_dim == 12
    assert env.model is not None
    assert env.data is not None


def test_mujoco_env_reset():
    """Verify reset contracts and observation bounds."""
    env = MuJoCoQuadrupedEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (48,)
    assert not np.isnan(obs).any()
    assert "cmd_vel" in info
    assert np.allclose(info["cmd_vel"], [0.5, 0.0, 0.0])


def test_mujoco_env_step():
    """Verify single step dynamics execution and reward return."""
    env = MuJoCoQuadrupedEnv()
    env.reset(seed=42)
    action = np.zeros(12)
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (48,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "base_height" in info
    assert info["base_height"] > 0.25


def test_mujoco_standing_stability():
    """Verify that zero action (nominal PD tracking) maintains stable posture without falling."""
    env = MuJoCoQuadrupedEnv(max_episode_steps=100)
    obs, _ = env.reset(seed=42)

    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(np.zeros(12))
        assert not terminated, f"Robot fell prematurely: height={info['base_height']}"

    assert info["base_height"] > 0.30, f"Robot sank too low: height={info['base_height']}"
    assert abs(info["base_roll"]) < 0.1
    assert abs(info["base_pitch"]) < 0.1
