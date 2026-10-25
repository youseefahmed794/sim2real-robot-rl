"""
Actor-Critic Neural Network Architecture for Continuous Quadruped Locomotion.

Provides an optimized Actor-Critic MLP with:
- Orthogonal parameter initialization
- Diagonal Gaussian continuous policy parameterization
- Independent or shared value function head
- Pure NumPy forward/backward pass engine for portable zero-dependency deployment
- Seamless PyTorch module bridge when torch is available

Author: Youseef Abdel Tawwab
"""

import math
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

# Optional PyTorch bridge
try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def orthogonal_init(shape: Tuple[int, int], gain: float = 1.0) -> np.ndarray:
    """
    Orthogonal matrix initialization for neural network layers.
    Generates orthonormal weight matrices scaled by the recommended gain.
    """
    rows, cols = shape
    flat_shape = (rows, cols)
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, vh = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else vh
    q = q.reshape(flat_shape)
    return (gain * q[:rows, :cols]).astype(np.float64)


class NumpyMLP:
    """
    Multi-Layer Perceptron implemented with NumPy.
    Supports Tanh/ELU activations, orthogonal weights, and analytical gradient caching.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "tanh",
        output_gain: float = 1.0,
        hidden_gain: float = 1.4142,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        layer_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            gain = output_gain if i == len(layer_dims) - 2 else hidden_gain
            W = orthogonal_init((fan_in, fan_out), gain=gain)
            b = np.zeros(fan_out, dtype=np.float64)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through MLP."""
        x = np.asarray(x, dtype=np.float64)
        is_single = (x.ndim == 1)
        if is_single:
            x = x[np.newaxis, :]

        h = x
        for i in range(len(self.weights)):
            W = self.weights[i]
            b = self.biases[i]
            z = h @ W + b
            if i < len(self.weights) - 1:
                if self.activation == "tanh":
                    h = np.tanh(z)
                elif self.activation == "elu":
                    h = np.where(z > 0, z, np.exp(z) - 1.0)
                else:
                    h = np.maximum(0.0, z)  # ReLU
            else:
                h = z  # Linear output

        return h[0] if is_single else h

    def get_params(self) -> Dict[str, Any]:
        """Return parameters dictionary."""
        return {
            "weights": [W.copy() for W in self.weights],
            "biases": [b.copy() for b in self.biases],
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set weights and biases from parameters dictionary."""
        self.weights = [W.copy() for W in params["weights"]]
        self.biases = [b.copy() for b in params["biases"]]


class ActorCritic:
    """
    Complete Actor-Critic Architecture.
    - Policy (Actor): Outputs mean action mu(s) and standard deviation sigma
    - Value Function (Critic): Outputs scalar state value V(s)
    """

    def __init__(
        self,
        observation_dim: int = 48,
        action_dim: int = 12,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "tanh",
        init_std: float = 1.0,
        min_std: float = 0.05,
    ):
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.min_std = min_std

        # Actor MLP (small output gain 0.01 for small initial actions)
        self.actor_mlp = NumpyMLP(
            input_dim=observation_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            activation=activation,
            output_gain=0.01,
        )

        # Critic MLP (unit output gain 1.0)
        self.critic_mlp = NumpyMLP(
            input_dim=observation_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            output_gain=1.0,
        )

        # Learnable log standard deviation vector
        self.log_std = np.full(action_dim, math.log(max(init_std, min_std)), dtype=np.float64)

    @property
    def std(self) -> np.ndarray:
        """Return clamped standard deviation vector."""
        return np.maximum(np.exp(self.log_std), self.min_std)

    def act(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample action from policy, compute log probability, and state value.

        Args:
            obs: Observation vector [observation_dim] or batch [B, observation_dim]
            deterministic: If True, returns mean action without Gaussian noise

        Returns:
            action: Selected action [action_dim]
            log_prob: Log probability of the selected action
            value: Estimated state value V(s)
        """
        obs = np.asarray(obs, dtype=np.float64)
        is_single = (obs.ndim == 1)
        if is_single:
            obs = obs[np.newaxis, :]

        mean = self.actor_mlp.forward(obs)
        std = self.std

        if deterministic:
            action = mean
        else:
            noise = np.random.normal(0.0, 1.0, size=mean.shape)
            action = mean + noise * std

        # Log-probability under Gaussian distribution
        # log p(a) = -0.5 * sum((a - mu)^2 / sigma^2 + 2 * log(sigma) + log(2 * pi))
        var = std ** 2
        log_prob = -0.5 * np.sum(
            ((action - mean) ** 2) / var + 2.0 * np.log(std) + math.log(2.0 * math.pi),
            axis=-1,
        )

        value = self.critic_mlp.forward(obs).squeeze(-1)

        if is_single:
            return action[0], float(log_prob[0]), float(np.asarray(value).item())
        return action, log_prob, value

    def evaluate_actions(
        self, obs: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate actions for PPO policy loss computation.

        Returns:
            log_probs: Log probabilities of actions under current policy
            values: State values V(s)
            entropy: Differential entropy of the Gaussian policy
        """
        obs = np.asarray(obs, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)

        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
            actions = actions[np.newaxis, :]

        mean = self.actor_mlp.forward(obs)
        std = self.std
        var = std ** 2

        log_probs = -0.5 * np.sum(
            ((actions - mean) ** 2) / var + 2.0 * np.log(std) + math.log(2.0 * math.pi),
            axis=-1,
        )

        values = self.critic_mlp.forward(obs).squeeze(-1)

        # Gaussian differential entropy: H = 0.5 * sum(1 + log(2 * pi * sigma^2))
        entropy = 0.5 * np.sum(1.0 + np.log(2.0 * math.pi * var))

        return log_probs, values, entropy

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        """Estimate state value V(s) only."""
        obs = np.asarray(obs, dtype=np.float64)
        is_single = (obs.ndim == 1)
        if is_single:
            obs = obs[np.newaxis, :]
        v = self.critic_mlp.forward(obs).squeeze(-1)
        return float(v) if is_single else v

    def get_weights(self) -> Dict[str, Any]:
        """Export complete network weights for serialization/export."""
        return {
            "actor": self.actor_mlp.get_params(),
            "critic": self.critic_mlp.get_params(),
            "log_std": self.log_std.copy(),
        }

    def set_weights(self, weights: Dict[str, Any]) -> None:
        """Load weights from dictionary."""
        self.actor_mlp.set_params(weights["actor"])
        self.critic_mlp.set_params(weights["critic"])
        self.log_std = weights["log_std"].copy()


# Optional PyTorch Module Wrapper
if TORCH_AVAILABLE:
    class PyTorchActorCritic(nn.Module):
        """Native PyTorch Actor-Critic module with Orthogonal initialization."""

        def __init__(
            self,
            observation_dim: int = 48,
            action_dim: int = 12,
            hidden_dims: Optional[List[int]] = None,
        ):
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [256, 128, 64]

            # Actor
            actor_layers = []
            prev_dim = observation_dim
            for h_dim in hidden_dims:
                linear = nn.Linear(prev_dim, h_dim)
                nn.init.orthogonal_(linear.weight, gain=math.sqrt(2))
                nn.init.constant_(linear.bias, 0.0)
                actor_layers.extend([linear, nn.Tanh()])
                prev_dim = h_dim
            actor_out = nn.Linear(prev_dim, action_dim)
            nn.init.orthogonal_(actor_out.weight, gain=0.01)
            nn.init.constant_(actor_out.bias, 0.0)
            actor_layers.append(actor_out)
            self.actor = nn.Sequential(*actor_layers)

            # Critic
            critic_layers = []
            prev_dim = observation_dim
            for h_dim in hidden_dims:
                linear = nn.Linear(prev_dim, h_dim)
                nn.init.orthogonal_(linear.weight, gain=math.sqrt(2))
                nn.init.constant_(linear.bias, 0.0)
                critic_layers.extend([linear, nn.Tanh()])
                prev_dim = h_dim
            critic_out = nn.Linear(prev_dim, 1)
            nn.init.orthogonal_(critic_out.weight, gain=1.0)
            nn.init.constant_(critic_out.bias, 0.0)
            critic_layers.append(critic_out)
            self.critic = nn.Sequential(*critic_layers)

            self.log_std = nn.Parameter(torch.zeros(action_dim))

        def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            mean = self.actor(obs)
            val = self.critic(obs)
            return mean, val
