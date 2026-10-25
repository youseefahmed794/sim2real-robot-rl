"""
Rollout Buffer with Generalized Advantage Estimation (GAE).

Stores trajectories collected from policy rollouts and computes:
- Generalized Advantage Estimators (GAE-Lambda)
- Discounted Monte Carlo / TD(lambda) returns
- Shuffled mini-batch generation for stochastic gradient updates

Author: Youseef Abdel Tawwab
"""

from typing import Generator, Dict, Any, Optional
import numpy as np


class RolloutBuffer:
    """
    On-Policy Rollout Buffer for PPO.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_dim: int = 48,
        action_dim: int = 12,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam

        # Preallocate continuous memory buffers
        self.observations = np.zeros((buffer_size, observation_dim), dtype=np.float64)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float64)
        self.rewards = np.zeros(buffer_size, dtype=np.float64)
        self.dones = np.zeros(buffer_size, dtype=bool)
        self.values = np.zeros(buffer_size, dtype=np.float64)
        self.log_probs = np.zeros(buffer_size, dtype=np.float64)
        self.advantages = np.zeros(buffer_size, dtype=np.float64)
        self.returns = np.zeros(buffer_size, dtype=np.float64)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Add a single step transition to the buffer."""
        if self.ptr >= self.buffer_size:
            raise IndexError("RolloutBuffer is full. Call compute_returns_and_advantages() and clear().")

        self.observations[self.ptr] = np.asarray(obs, dtype=np.float64)
        self.actions[self.ptr] = np.asarray(action, dtype=np.float64)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = bool(done)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float = 0.0) -> None:
        """
        Compute Generalized Advantage Estimation (GAE) and TD(lambda) returns.

        Recursive backward computation:
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = delta_t + (gamma * lam) * (1 - done_t) * A_{t+1}
            R_t = A_t + V(s_t)
        """
        last_gae = 0.0
        n_steps = self.ptr

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = float(last_value)
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        # Returns R_t = A_t + V(s_t)
        self.returns[:n_steps] = self.advantages[:n_steps] + self.values[:n_steps]

    def get_generator(
        self, batch_size: int, normalize_adv: bool = True
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """
        Yield randomized mini-batches for PPO update epochs.
        """
        n_steps = self.ptr
        indices = np.random.permutation(n_steps)

        # Normalize advantages across the buffer
        adv = self.advantages[:n_steps]
        if normalize_adv and n_steps > 1:
            mean = np.mean(adv)
            std = np.std(adv) + 1e-8
            adv_normalized = (adv - mean) / std
        else:
            adv_normalized = adv.copy()

        for start_idx in range(0, n_steps, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield {
                "observations": self.observations[batch_indices],
                "actions": self.actions[batch_indices],
                "values": self.values[batch_indices],
                "log_probs": self.log_probs[batch_indices],
                "advantages": adv_normalized[batch_indices],
                "returns": self.returns[batch_indices],
            }

    def clear(self) -> None:
        """Reset buffer pointers."""
        self.ptr = 0
        self.full = False
