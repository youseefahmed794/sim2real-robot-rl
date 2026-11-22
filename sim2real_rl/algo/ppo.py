"""
Proximal Policy Optimization (PPO) with Clipped Objective and Generalized Advantage Estimation.

Implements the standard clipped surrogate objective:
    L_CLIP(theta) = E [ min(r_t * A_t, clip(r_t, 1 - eps, 1 + eps) * A_t) ]
Combined with value function loss, entropy regularization, and KL divergence early stopping.

Author: Youseef Abdel Tawwab
"""

import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional
import numpy as np

from sim2real_rl.algo import network
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
    """Differentiable PPO training; NumPy networks are inference-only."""

    def __init__(self, actor_critic=None, config=None, observation_dim=48, action_dim=12):
        if not network.TORCH_AVAILABLE:
            raise ImportError('PPO training requires torch; install .[training]')
        self.config = config or PPOConfig()
        self.actor_critic = actor_critic if actor_critic is not None else network.PyTorchActorCritic(
            observation_dim=observation_dim, action_dim=action_dim)
        if not isinstance(self.actor_critic, network.PyTorchActorCritic):
            raise TypeError('PPO requires PyTorchActorCritic; ActorCritic is NumPy inference-only')
        self._make_optimizer()
        self.update_count = 0

    def _make_optimizer(self):
        self.lr = self.config.learning_rate
        self.optimizer = network.torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr)

    def _loss_tensors(self, obs, actions, old_log_probs, advantages, returns):
        torch = network.torch
        logp, values, entropy = self.actor_critic.evaluate_actions(obs, actions)
        tensor = self.actor_critic._tensor
        log_ratio = logp - tensor(old_log_probs)
        ratio = log_ratio.exp()
        adv = tensor(advantages)
        policy_loss = -torch.minimum(ratio * adv, ratio.clamp(
            1-self.config.clip_param, 1+self.config.clip_param) * adv).mean()
        value_loss = (values - tensor(returns)).square().mean()
        return {
            "total_loss": policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy,
            "policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy,
            "approx_kl": ((ratio - 1) - log_ratio).mean(),
            "clip_fraction": ((ratio - 1).abs() > self.config.clip_param).float().mean(),
        }

    def compute_losses(self, obs, actions, old_log_probs, advantages, returns):
        """Detached metrics API; train_step uses the same differentiable objective."""
        with network.torch.no_grad():
            return {k: float(v) for k, v in self._loss_tensors(
                obs, actions, old_log_probs, advantages, returns).items()}

    def train_step(self, buffer: RolloutBuffer) -> Dict[str, float]:
        torch = network.torch
        if not buffer.ptr:
            raise ValueError('Cannot update PPO with an empty rollout')
        metrics = {}
        steps = 0
        stopped = False
        for _ in range(self.config.num_epochs):
            for batch in buffer.get_generator(self.config.mini_batch_size):
                losses = self._loss_tensors(batch['observations'], batch['actions'], batch['log_probs'],
                                           batch['advantages'], batch['returns'])
                if not all(torch.isfinite(v).all() for v in losses.values()):
                    raise FloatingPointError('Nonfinite PPO objective')
                for k, v in losses.items():
                    metrics.setdefault(k, []).append(float(v.detach()))
                if self.config.target_kl is not None and float(losses['approx_kl'].detach()) > 1.5*self.config.target_kl:
                    stopped = True
                    break
                self.optimizer.zero_grad(set_to_none=True)
                losses['total_loss'].backward()
                for group in ('actor', 'critic', 'log_std'):
                    params = [self.actor_critic.log_std] if group == 'log_std' else list(getattr(self.actor_critic, group).parameters())
                    norm = torch.sqrt(sum(p.grad.square().sum() for p in params if p.grad is not None))
                    metrics.setdefault(group+'_grad_norm', []).append(float(norm))
                norm = torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm,
                                                     error_if_nonfinite=True)
                metrics.setdefault('grad_norm', []).append(float(norm))
                self.optimizer.step()
                self.update_count += 1
                steps += 1
            if stopped:
                break
        result = {k: float(np.mean(v)) for k, v in metrics.items()}
        result.update(optimizer_steps=steps, early_stopped=stopped)
        return result

    def save(self, filepath):
        """Trusted local checkpoint, including Adam moments and sampling RNG state."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        network.torch.save(dict(format_version=1, backend='torch',
            architecture=self.actor_critic.architecture(), config=asdict(self.config),
            model=self.actor_critic.state_dict(), optimizer=self.optimizer.state_dict(),
            update_count=self.update_count, numpy_rng=np.random.get_state(),
            torch_rng=network.torch.get_rng_state()), filepath)

    def load(self, filepath, restore_rng=True):
        """Restore a trusted training checkpoint. Old NumPy pickles remain inference artifacts."""
        checkpoint = network.torch.load(filepath, map_location='cpu', weights_only=False)
        if checkpoint.get('backend') != 'torch' or checkpoint.get('format_version') != 1:
            raise ValueError('Expected a version 1 torch PPO checkpoint')
        self.actor_critic = network.PyTorchActorCritic(**checkpoint['architecture'])
        self.actor_critic.load_state_dict(checkpoint['model'])
        self.config = PPOConfig(**checkpoint['config'])
        self._make_optimizer()
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint['update_count']
        if restore_rng:
            np.random.set_state(checkpoint['numpy_rng'])
            network.torch.set_rng_state(checkpoint['torch_rng'])
