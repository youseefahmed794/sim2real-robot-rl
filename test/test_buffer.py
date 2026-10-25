"""
Unit tests for Rollout Buffer and GAE Advantage Computation.

Author: Youseef Abdel Tawwab
"""

import numpy as np
import pytest

from sim2real_rl.algo.buffer import RolloutBuffer


def test_buffer_add_and_overflow():
    """Verify transition insertion and boundary checks."""
    buf = RolloutBuffer(buffer_size=10, observation_dim=48, action_dim=12)

    for i in range(10):
        buf.add(
            obs=np.zeros(48),
            action=np.zeros(12),
            reward=1.0,
            done=(i == 9),
            value=0.5,
            log_prob=-1.2,
        )

    assert buf.full
    with pytest.raises(IndexError):
        buf.add(np.zeros(48), np.zeros(12), 1.0, False, 0.5, -1.2)


def test_gae_computation_correctness():
    """Verify Generalized Advantage Estimation against analytical 1-step Bellman equation."""
    buf = RolloutBuffer(buffer_size=3, observation_dim=4, action_dim=2, gamma=0.9, lam=0.8)

    # Transition 0: r=1.0, v=2.0, done=False
    buf.add(np.zeros(4), np.zeros(2), reward=1.0, done=False, value=2.0, log_prob=0.0)
    # Transition 1: r=2.0, v=3.0, done=False
    buf.add(np.zeros(4), np.zeros(2), reward=2.0, done=False, value=3.0, log_prob=0.0)
    # Transition 2: r=0.5, v=1.5, done=True
    buf.add(np.zeros(4), np.zeros(2), reward=0.5, done=True, value=1.5, log_prob=0.0)

    # Compute advantages with last_value = 0.0
    buf.compute_returns_and_advantages(last_value=0.0)

    # Analytical verification:
    # t=2 (done=True): delta_2 = 0.5 + 0.9*0*(0) - 1.5 = -1.0. A_2 = -1.0. R_2 = -1.0 + 1.5 = 0.5
    # t=1: delta_1 = 2.0 + 0.9*1.5*(1) - 3.0 = 2.0 + 1.35 - 3.0 = 0.35. A_1 = 0.35 + (0.9*0.8)*1*(-1.0) = 0.35 - 0.72 = -0.37
    # t=0: delta_0 = 1.0 + 0.9*3.0*(1) - 2.0 = 1.0 + 2.7 - 2.0 = 1.7. A_0 = 1.7 + (0.72)*1*(-0.37) = 1.7 - 0.2664 = 1.4336
    assert np.isclose(buf.advantages[2], -1.0, atol=1e-5)
    assert np.isclose(buf.returns[2], 0.5, atol=1e-5)
    assert np.isclose(buf.advantages[1], -0.37, atol=1e-5)
    assert np.isclose(buf.advantages[0], 1.4336, atol=1e-5)


def test_buffer_minibatch_generator():
    """Verify minibatch generator yields all transitions and handles normalization."""
    buf = RolloutBuffer(buffer_size=20, observation_dim=48, action_dim=12)
    for _ in range(20):
        buf.add(
            obs=np.random.randn(48),
            action=np.random.randn(12),
            reward=np.random.rand(),
            done=False,
            value=np.random.rand(),
            log_prob=-1.0,
        )
    buf.compute_returns_and_advantages(last_value=0.0)

    total_samples = 0
    for batch in buf.get_generator(batch_size=8, normalize_adv=True):
        total_samples += len(batch["observations"])
        assert batch["observations"].shape[-1] == 48
        assert batch["actions"].shape[-1] == 12

    assert total_samples == 20
