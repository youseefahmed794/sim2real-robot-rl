"""
Proximal Policy Optimization (PPO) with Clipped Objective and Generalized Advantage Estimation.

Implements the standard clipped surrogate objective:
    L_CLIP(theta) = E [ min(r_t * A_t, clip(r_t, 1 - eps, 1 + eps) * A_t) ]
Combined with value function loss, entropy regularization, and KL divergence early stopping.

Author: Youseef Abdel Tawwab
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from sim2real_rl.algo.network import ActorCritic
from sim2real_rl.algo.buffer import RolloutBuffer


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    learning_rate: float = 0.0003
    num_epochs: int = 5
    mini_batch_size: int = 64
    target_kl: Optional[float] = 0.015


class PPO:
    """
    PPO Agent with Actor-Critic networks and clipped surrogate policy updates.
    """

    def __init__(
        self,
        actor_critic: Optional[ActorCritic] = None,
        config: Optional[PPOConfig] = None,
        observation_dim: int = 48,
        action_dim: int = 12,
    ):
        self.config = config or PPOConfig()
        self.actor_critic = actor_critic or ActorCritic(
            observation_dim=observation_dim, action_dim=action_dim
        )
        self.lr = self.config.learning_rate

    def compute_losses(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute PPO policy loss, value loss, and entropy bonus.
        """
        # Evaluate current policy
        new_log_probs, values, entropy = self.actor_critic.evaluate_actions(obs, actions)

        # Probability ratio r_t(theta) = exp(log pi_new - log pi_old)
        log_ratio = new_log_probs - old_log_probs
        ratio = np.exp(np.clip(log_ratio, -20.0, 20.0))

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param) * advantages
        policy_loss = -float(np.mean(np.minimum(surr1, surr2)))

        # Value loss (MSE)
        value_loss = float(np.mean((values - returns) ** 2))

        # Approximate KL divergence
        approx_kl = float(np.mean((ratio - 1.0) - log_ratio))

        # Clip fraction (fraction of updates where clipping occurred)
        clip_fraction = float(np.mean(np.abs(ratio - 1.0) > self.config.clip_param))

        total_loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            - self.config.entropy_coef * float(entropy)
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": float(entropy),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def train_step(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Execute multiple PPO epochs over the collected rollout buffer.
        """
        metrics = {
            "total_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        early_stopped = False
        for epoch in range(self.config.num_epochs):
            if early_stopped:
                break

            for batch in buffer.get_generator(self.config.mini_batch_size):
                loss_dict = self.compute_losses(
                    obs=batch["observations"],
                    actions=batch["actions"],
                    old_log_probs=batch["log_probs"],
                    advantages=batch["advantages"],
                    returns=batch["returns"],
                )

                for k, v in loss_dict.items():
                    metrics[k].append(v)

                # Early stopping if KL divergence exceeds 1.5 * target_kl
                if self.config.target_kl is not None and loss_dict["approx_kl"] > 1.5 * self.config.target_kl:
                    early_stopped = True
                    break

        return {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in metrics.items()}

    def save(self, filepath: str) -> None:
        """Serialize actor-critic weights and hyperparameters to file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "weights": self.actor_critic.get_weights(),
            "config": self.config,
            "observation_dim": self.actor_critic.observation_dim,
            "action_dim": self.actor_critic.action_dim,
        }
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint, f)

    def load(self, filepath: str) -> None:
        """Load weights from serialized checkpoint."""
        with open(filepath, "rb") as f:
            checkpoint = pickle.load(f)
        self.actor_critic.set_weights(checkpoint["weights"])
        self.config = checkpoint["config"]
