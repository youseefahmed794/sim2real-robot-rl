"""
Actor-Critic Neural Network Architecture for Continuous Quadruped Locomotion.

Provides an optimized Actor-Critic MLP with:
- Orthogonal parameter initialization
- Diagonal Gaussian continuous policy parameterization
- Independent or shared value function head
- NumPy forward-only inference for portable deployment
- Native differentiable PyTorch training backend (required by PPO)

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
    Forward-only inference with Tanh/ELU activations and orthogonal weights.
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
    NumPy inference-only Actor-Critic Architecture (not a training backend).
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


# PyTorch is optional for NumPy inference, mandatory for PPO training.
if TORCH_AVAILABLE:
    class PyTorchActorCritic(nn.Module):
        """Trainable latent Gaussian policy. Environment clipping stays outside it."""

        def __init__(self, observation_dim=48, action_dim=12, hidden_dims=None,
                     activation="tanh", init_std=1.0, min_std=0.05):
            super().__init__()
            self.observation_dim = observation_dim
            self.action_dim = action_dim
            self.hidden_dims = [256, 128, 64] if hidden_dims is None else list(hidden_dims)
            self.activation = activation
            self.min_std = min_std
            if min_std <= 0 or init_std <= 0:
                raise ValueError("Standard deviations must be positive")
            activations = {"tanh": nn.Tanh, "elu": nn.ELU, "relu": nn.ReLU}
            if activation not in activations:
                raise ValueError(f"Unknown activation: {activation}")

            def mlp(output_dim, gain):
                layers = []
                dims = [observation_dim] + self.hidden_dims + [output_dim]
                for i, (din, dout) in enumerate(zip(dims, dims[1:])):
                    linear = nn.Linear(din, dout)
                    last = i == len(dims) - 2
                    nn.init.orthogonal_(linear.weight, gain=gain if last else math.sqrt(2))
                    nn.init.zeros_(linear.bias)
                    layers.append(linear)
                    if not last:
                        layers.append(activations[activation]())
                return nn.Sequential(*layers)

            self.actor = mlp(action_dim, 0.01)
            self.critic = mlp(1, 1.0)
            self.log_std = nn.Parameter(torch.full((action_dim,), math.log(max(init_std, min_std))))

        @property
        def std(self):
            # Identical floor in sampling, likelihood and entropy. Below the floor
            # the scale gradient is zero, as required by this parameterization.
            return self.log_std.clamp_min(math.log(self.min_std)).exp()

        def forward(self, obs):
            return self.actor(obs), self.critic(obs).squeeze(-1)

        def _tensor(self, x):
            return torch.as_tensor(x, dtype=self.log_std.dtype, device=self.log_std.device)

        def act(self, obs, deterministic=False):
            with torch.no_grad():
                mean, value = self.forward(self._tensor(obs))
                dist = Normal(mean, self.std)
                action = mean if deterministic else dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
                return tuple(t.cpu().numpy() for t in (action, log_prob, value))

        def get_value(self, obs):
            with torch.no_grad():
                return self.critic(self._tensor(obs)).squeeze(-1).cpu().numpy()

        def evaluate_actions(self, obs, actions):
            mean, value = self.forward(self._tensor(obs))
            dist = Normal(mean, self.std)
            return (dist.log_prob(self._tensor(actions)).sum(-1), value,
                    dist.entropy().sum(-1).mean())

        def architecture(self):
            return dict(observation_dim=self.observation_dim, action_dim=self.action_dim,
                        hidden_dims=self.hidden_dims, activation=self.activation,
                        min_std=self.min_std)

        def get_weights(self):
            """Export NumPy-compatible inference weights; training owns the tensors."""
            result = {}
            for name in ("actor", "critic"):
                layers = [m for m in getattr(self, name) if isinstance(m, nn.Linear)]
                result[name] = {
                    "weights": [m.weight.detach().cpu().numpy().T.copy() for m in layers],
                    "biases": [m.bias.detach().cpu().numpy().copy() for m in layers]}
            result["log_std"] = self.log_std.detach().cpu().numpy().copy()
            return result

        def set_weights(self, weights):
            with torch.no_grad():
                for name in ("actor", "critic"):
                    layers = [m for m in getattr(self, name) if isinstance(m, nn.Linear)]
                    for layer, w, b in zip(layers, weights[name]["weights"], weights[name]["biases"]):
                        layer.weight.copy_(self._tensor(w.T))
                        layer.bias.copy_(self._tensor(b))
                self.log_std.copy_(self._tensor(weights["log_std"]))
