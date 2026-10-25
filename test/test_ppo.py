"""
Unit tests for PPO Agent and Loss Optimization.

Author: Youseef Abdel Tawwab
"""

import os
import tempfile
import numpy as np
import pytest

from sim2real_rl.algo.network import ActorCritic
from sim2real_rl.algo.buffer import RolloutBuffer
from sim2real_rl.algo.ppo import PPO, PPOConfig


def test_ppo_loss_computation():
    """Verify clipped surrogate loss and value loss computation."""
    ppo = PPO(config=PPOConfig(clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01))

    N = 16
    obs = np.random.randn(N, 48)
    actions = np.random.randn(N, 12)
    old_log_probs = np.full(N, -5.0)
    advantages = np.ones(N)
    returns = np.zeros(N)

    losses = ppo.compute_losses(obs, actions, old_log_probs, advantages, returns)
    assert "total_loss" in losses
    assert "policy_loss" in losses
    assert "value_loss" in losses
    assert "approx_kl" in losses
    assert "clip_fraction" in losses
    assert losses["value_loss"] >= 0.0


def test_ppo_clipping_activation():
    """Verify that when action probability ratio deviates far from 1, clipping activates."""
    ppo = PPO(config=PPOConfig(clip_param=0.2))

    N = 10
    obs = np.zeros((N, 48))
    actions = np.zeros((N, 12))
    # Artificially set very different old log probs to force large ratio
    old_log_probs = np.full(N, -20.0)
    advantages = np.ones(N)
    returns = np.zeros(N)

    losses = ppo.compute_losses(obs, actions, old_log_probs, advantages, returns)
    assert losses["clip_fraction"] > 0.5, "Clipping should be active when ratio is large"


def test_ppo_train_step_execution():
    """Verify complete train_step execution on a simulated rollout buffer."""
    ppo = PPO(config=PPOConfig(num_epochs=2, mini_batch_size=8))
    buf = RolloutBuffer(buffer_size=16, observation_dim=48, action_dim=12)

    for _ in range(16):
        buf.add(
            obs=np.random.randn(48),
            action=np.random.randn(12),
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=-2.0,
        )
    buf.compute_returns_and_advantages(last_value=0.5)

    metrics = ppo.train_step(buf)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics


def test_ppo_save_and_load():
    """Verify agent checkpoint serialization and deserialization."""
    ppo = PPO(config=PPOConfig(learning_rate=0.0005))
    obs = np.random.randn(48)
    act_orig, _, _ = ppo.actor_critic.act(obs, deterministic=True)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        path = tmp.name

    try:
        ppo.save(path)
        assert os.path.exists(path)

        ppo_loaded = PPO()
        ppo_loaded.load(path)

        act_loaded, _, _ = ppo_loaded.actor_critic.act(obs, deterministic=True)
        assert np.allclose(act_orig, act_loaded)
        assert ppo_loaded.config.learning_rate == 0.0005
    finally:
        if os.path.exists(path):
            os.remove(path)
