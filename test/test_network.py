"""
Unit tests for Actor-Critic Neural Network Architecture.

Author: Youseef Abdel Tawwab
"""

import numpy as np
import pytest

from sim2real_rl.algo.network import ActorCritic, NumpyMLP, orthogonal_init


def test_orthogonal_init():
    """Verify orthogonal matrix initialization properties."""
    W = orthogonal_init((64, 64), gain=1.0)
    assert np.allclose(W @ W.T, np.eye(64), atol=1e-5)


def test_numpy_mlp_forward():
    """Verify forward inference of pure NumPy MLP."""
    mlp = NumpyMLP(input_dim=48, hidden_dims=[64, 32], output_dim=12)
    x_single = np.ones(48)
    out_single = mlp.forward(x_single)
    assert out_single.shape == (12,)

    x_batch = np.ones((16, 48))
    out_batch = mlp.forward(x_batch)
    assert out_batch.shape == (16, 12)


def test_actor_critic_shapes_and_stochasticity():
    """Verify Actor-Critic action sampling, log-prob, and value estimation shapes."""
    ac = ActorCritic(observation_dim=48, action_dim=12)

    obs_single = np.random.randn(48)
    action, log_prob, val = ac.act(obs_single, deterministic=False)
    assert action.shape == (12,)
    assert isinstance(log_prob, (float, np.floating))
    assert isinstance(val, (float, np.floating))

    # Deterministic should equal mean
    action_det, _, _ = ac.act(obs_single, deterministic=True)
    action_det2, _, _ = ac.act(obs_single, deterministic=True)
    assert np.allclose(action_det, action_det2)

    # Batch test
    obs_batch = np.random.randn(8, 48)
    actions, log_probs, values = ac.act(obs_batch)
    assert actions.shape == (8, 12)
    assert log_probs.shape == (8,)
    assert values.shape == (8,)


def test_actor_critic_evaluation_and_entropy():
    """Verify action evaluation and Gaussian differential entropy computation."""
    ac = ActorCritic(observation_dim=48, action_dim=12)
    obs = np.random.randn(10, 48)
    actions = np.random.randn(10, 12)

    log_probs, values, entropy = ac.evaluate_actions(obs, actions)
    assert log_probs.shape == (10,)
    assert values.shape == (10,)
    assert entropy > 0.0, "Entropy of continuous Gaussian should be positive"


def test_actor_critic_weights_serialization():
    """Verify weight export and restore roundtrip."""
    ac1 = ActorCritic(observation_dim=48, action_dim=12)
    obs = np.random.randn(48)
    a1, _, v1 = ac1.act(obs, deterministic=True)

    weights = ac1.get_weights()
    ac2 = ActorCritic(observation_dim=48, action_dim=12)
    ac2.set_weights(weights)

    a2, _, v2 = ac2.act(obs, deterministic=True)
    assert np.allclose(a1, a2)
    assert np.isclose(v1, v2)
