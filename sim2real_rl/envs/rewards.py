"""
Multi-Objective Reward Formulation for Dynamic Quadruped Locomotion.
Decomposes task rewards into tracking velocity objectives, body stability regularization,
actuator smoothness penalties, and energy minimization terms.

Author: Youseef Abdel Tawwab
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class RewardConfig:
    # Tracking terms (positive exponential kernels)
    w_lin_vel: float = 1.5           # Linear velocity tracking weight
    w_ang_vel: float = 0.8           # Angular velocity tracking weight
    sigma_lin_vel: float = 0.25      # Bandwidth for linear velocity exp kernel (m/s)
    sigma_ang_vel: float = 0.50      # Bandwidth for angular velocity exp kernel (rad/s)

    # Base stability penalties (negative quadratic)
    w_lin_vel_z: float = -2.0        # Penalize vertical bouncing (v_z^2)
    w_ang_vel_xy: float = -0.05      # Penalize roll and pitch rates (omega_x^2 + omega_y^2)
    w_orientation: float = -1.0      # Penalize tilt from upright (||g_proj_xy||^2)
    w_base_height: float = -10.0     # Penalize base height error (z - z_nom)^2
    nominal_height: float = 0.28     # Desired base height (m)

    # Energy & Smoothness penalties
    w_torques: float = -1e-4         # Penalize sum(tau_i^2)
    w_mechanical_power: float = -2e-4# Penalize sum(|tau_i * dq_i|)
    w_action_rate: float = -0.015    # Penalize ||a_t - a_{t-1}||^2
    w_joint_acc: float = -2.5e-7     # Penalize ||ddq||^2

    # Kinematic & Safety penalties
    w_joint_limits: float = -10.0    # Penalize exceeding joint limits
    w_foot_slip: float = -0.04       # Penalize tangential foot velocity during stance

    # Survival / alive reward
    w_alive: float = 1.0             # Constant bonus per non-terminated step


class RewardManager:
    """
    Manages computation, decomposition, and telemetry logging
    of the multi-objective reward formulation for quadruped locomotion.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config if config is not None else RewardConfig()
        self.prev_action = np.zeros(12, dtype=np.float64)
        self.prev_joint_vel = np.zeros(12, dtype=np.float64)
        self.last_reward_breakdown: Dict[str, float] = {}

    def reset(self) -> None:
        """Reset internal buffers on episode restart."""
        self.prev_action = np.zeros(12, dtype=np.float64)
        self.prev_joint_vel = np.zeros(12, dtype=np.float64)
        self.last_reward_breakdown.clear()

    def compute_reward(
        self,
        base_lin_vel_base: np.ndarray,      # [vx, vy, vz] in base frame
        base_ang_vel_base: np.ndarray,      # [wx, wy, wz] in base frame
        projected_gravity: np.ndarray,      # [gx, gy, gz] in base frame
        cmd_vel: np.ndarray,                # [vx_cmd, vy_cmd, wz_cmd]
        base_pos_z: float,                  # current base z elevation
        joint_pos: np.ndarray,              # [12] joint positions (rad)
        joint_vel: np.ndarray,              # [12] joint velocities (rad/s)
        joint_torques: np.ndarray,          # [12] applied joint torques (N*m)
        current_action: np.ndarray,         # [12] current normalized action
        foot_contacts: np.ndarray,          # [4] bool contact states
        foot_velocities: np.ndarray,        # [4, 3] foot linear velocities in world frame
        joint_limits_lower: np.ndarray,     # [12] joint lower bounds
        joint_limits_upper: np.ndarray,     # [12] joint upper bounds
        dt: float = 0.02                    # control timestep (s)
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes composite scalar reward and returns detailed breakdown.
        """
        cfg = self.config
        breakdown: Dict[str, float] = {}

        # 1. Linear velocity tracking in base horizontal plane (vx, vy)
        lin_vel_xy_err = np.sum((base_lin_vel_base[:2] - cmd_vel[:2]) ** 2)
        r_lin_vel = np.exp(-lin_vel_xy_err / (cfg.sigma_lin_vel ** 2))
        breakdown["rew_lin_vel"] = float(cfg.w_lin_vel * r_lin_vel)

        # 2. Angular velocity tracking around base vertical axis (wz)
        ang_vel_z_err = (base_ang_vel_base[2] - cmd_vel[2]) ** 2
        r_ang_vel = np.exp(-ang_vel_z_err / (cfg.sigma_ang_vel ** 2))
        breakdown["rew_ang_vel"] = float(cfg.w_ang_vel * r_ang_vel)

        # 3. Base vertical velocity penalty (v_z^2)
        breakdown["pen_lin_vel_z"] = float(cfg.w_lin_vel_z * (base_lin_vel_base[2] ** 2))

        # 4. Base roll/pitch angular velocity penalty (wx^2 + wy^2)
        breakdown["pen_ang_vel_xy"] = float(cfg.w_ang_vel_xy * np.sum(base_ang_vel_base[:2] ** 2))

        # 5. Base orientation penalty: projected gravity xy components should be 0 when upright
        # When perfectly upright, projected gravity is [0, 0, -9.81], so gx^2 + gy^2 = 0
        tilt_err = np.sum(projected_gravity[:2] ** 2) / (9.81 ** 2)
        breakdown["pen_orientation"] = float(cfg.w_orientation * tilt_err)

        # 6. Base height tracking error
        height_err = (base_pos_z - cfg.nominal_height) ** 2
        breakdown["pen_base_height"] = float(cfg.w_base_height * height_err)

        # 7. Motor torque penalty (effort)
        torque_cost = np.sum(joint_torques ** 2)
        breakdown["pen_torques"] = float(cfg.w_torques * torque_cost)

        # 8. Mechanical power penalty
        power_cost = np.sum(np.abs(joint_torques * joint_vel))
        breakdown["pen_mechanical_power"] = float(cfg.w_mechanical_power * power_cost)

        # 9. Action rate penalty (smoothness)
        action_diff = current_action - self.prev_action
        breakdown["pen_action_rate"] = float(cfg.w_action_rate * np.sum(action_diff ** 2))

        # 10. Joint acceleration penalty
        joint_acc = (joint_vel - self.prev_joint_vel) / max(dt, 1e-4)
        breakdown["pen_joint_acc"] = float(cfg.w_joint_acc * np.sum(joint_acc ** 2))

        # 11. Soft joint limits penalty
        lower_violation = np.maximum(0.0, (joint_limits_lower + 0.05) - joint_pos)
        upper_violation = np.maximum(0.0, joint_pos - (joint_limits_upper - 0.05))
        joint_limit_cost = np.sum(lower_violation ** 2 + upper_violation ** 2)
        breakdown["pen_joint_limits"] = float(cfg.w_joint_limits * joint_limit_cost)

        # 12. Foot slip penalty (tangential velocity during ground contact)
        slip_cost = 0.0
        for i in range(4):
            if foot_contacts[i]:
                # Tangential velocity in ground plane
                v_xy_foot = foot_velocities[i, :2]
                slip_cost += np.sum(v_xy_foot ** 2)
        breakdown["pen_foot_slip"] = float(cfg.w_foot_slip * slip_cost)

        # 13. Alive bonus
        breakdown["rew_alive"] = float(cfg.w_alive)

        # Update historical state buffers
        self.prev_action = np.copy(current_action)
        self.prev_joint_vel = np.copy(joint_vel)
        self.last_reward_breakdown = breakdown

        total_reward = sum(breakdown.values())
        return float(total_reward), breakdown
