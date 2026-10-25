"""
Unit and Integration tests for Gymnasium QuadrupedLocomotion-v0 Environment.

Author: Youseef Abdel Tawwab
"""

import numpy as np
import pytest
try:
    import gymnasium as gym
except ImportError:
    gym = None

from sim2real_rl.envs.quadruped_env import QuadrupedEnv
from sim2real_rl.envs.terrain import TerrainConfig, TerrainType


def test_env_spaces():
    """Verify observation and action spaces match theoretical MDP dimensions."""
    env = QuadrupedEnv()

    assert env.action_space.shape == (12,)
    np.testing.assert_allclose(env.action_space.low, -1.0)
    np.testing.assert_allclose(env.action_space.high, 1.0)

    assert env.observation_space.shape == (48,)


def test_env_reset():
    """Verify reset produces a valid 48-dim float32 observation and info dict."""
    env = QuadrupedEnv()
    obs, info = env.reset(seed=123)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (48,)
    assert obs.dtype == np.float32
    assert not np.isnan(obs).any()
    assert not np.isinf(obs).any()

    assert "cmd_vel" in info
    assert len(info["cmd_vel"]) == 3
    assert "base_pos" in info


def test_env_step_contract():
    """Verify standard Gymnasium 5-tuple return format from step()."""
    env = QuadrupedEnv()
    env.reset(seed=42)

    zero_action = np.zeros(12, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(zero_action)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (48,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "reward_breakdown" in info
    assert "rew_alive" in info["reward_breakdown"]


def test_env_rollout_stability():
    """Verify multi-step rollout without numerical divergence (no NaN/Inf)."""
    env = QuadrupedEnv()
    obs, _ = env.reset(seed=99)

    for step_i in range(50):
        # Apply small periodic oscillating action
        action = 0.2 * np.sin(0.1 * step_i * np.ones(12, dtype=np.float32))
        obs, reward, terminated, truncated, info = env.step(action)

        assert not np.isnan(obs).any(), f"NaN detected in obs at step {step_i}"
        assert not np.isinf(obs).any(), f"Inf detected in obs at step {step_i}"
        assert not np.isnan(reward), f"NaN reward at step {step_i}"

        if terminated or truncated:
            break


def test_env_truncation():
    """Verify episode truncates when reaching max_episode_steps."""
    max_steps = 15
    env = QuadrupedEnv(max_episode_steps=max_steps)
    env.reset(seed=1)

    for i in range(max_steps - 1):
        _, _, terminated, truncated, _ = env.step(np.zeros(12))
        if terminated:
            return
        assert not truncated

    # Exactly at max_steps
    _, _, _, truncated, _ = env.step(np.zeros(12))
    assert truncated


def test_gymnasium_registry():
    """Verify environment registration and instantiation through gym.make."""
    if gym is None:
        pytest.skip("Gymnasium not installed")

    import sim2real_rl.envs  # Triggers registration
    env = gym.make("QuadrupedLocomotion-v0")
    obs, info = env.reset()
    assert obs.shape == (48,)
    env.close()
