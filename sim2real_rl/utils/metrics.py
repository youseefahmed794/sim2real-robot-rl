"""
Metrics and Scientific Evaluation Functions for Quadruped Locomotion.

Provides strictly defined formulas for:
- Mechanical power: P_mech = sum_i max(tau_i * dq_i, 0)
- Cost of Transport (CoT): CoT = P_mech / (m * g * v)
- Ground contact and survival verification

Author: Youseef Abdel Tawwab
"""

import math
from typing import Optional
import numpy as np


def compute_mechanical_power(torques: np.ndarray, joint_velocities: np.ndarray) -> float:
    """
    Computes instantaneous positive mechanical power dissipated by the actuators.

    Definition:
        P_mech = sum_{i=1}^{N} max(tau_i * dq_i, 0.0)

    Assumes non-regenerative actuation (quadruped BLDC motor drives dissipate negative
    work as thermal energy rather than recovering it to the DC bus).

    Args:
        torques: Array of joint torques (Nm), shape (N,).
        joint_velocities: Array of joint angular velocities (rad/s), shape (N,).

    Returns:
        float: Instantaneous positive mechanical power (Watts), >= 0.0.
    """
    torques = np.asarray(torques, dtype=np.float64)
    joint_velocities = np.asarray(joint_velocities, dtype=np.float64)
    power_per_joint = torques * joint_velocities
    positive_power = np.maximum(power_per_joint, 0.0)
    return float(np.sum(positive_power))


def compute_cost_of_transport(
    mean_mechanical_power: float,
    total_mass: float,
    gravity: float,
    mean_forward_velocity: float,
    min_velocity_threshold: float = 0.05,
) -> float:
    """
    Computes dimensionless Cost of Transport (CoT).

    Definition:
        CoT = P_mech / (m * g * v)

    Where:
        P_mech: Mean positive mechanical power (Watts)
        m: Total robot mass (kg)
        g: Gravitational acceleration (m/s^2)
        v: Mean forward speed (m/s)

    Handling stationary cases:
        When forward velocity v < min_velocity_threshold (default 0.05 m/s),
        returns float('nan') to avoid unphysical infinite or inflated values
        when the robot is not actually transporting mass forward.

    Args:
        mean_mechanical_power: Mean positive mechanical power over rollout (Watts).
        total_mass: Robot mass (kg), must be > 0.
        gravity: Gravitational acceleration (m/s^2), must be > 0.
        mean_forward_velocity: Mean forward speed in direction of command (m/s).
        min_velocity_threshold: Minimum velocity (m/s) required to compute CoT.

    Returns:
        float: Cost of transport (dimensionless), or float('nan') if velocity < threshold.
    """
    if total_mass <= 0.0 or gravity <= 0.0:
        raise ValueError(f"Mass ({total_mass}) and gravity ({gravity}) must be positive.")

    if mean_forward_velocity < min_velocity_threshold or math.isnan(mean_forward_velocity):
        return float("nan")

    cot = mean_mechanical_power / (total_mass * gravity * mean_forward_velocity)
    return float(cot)


def evaluate_survival(
    has_torso_contact: bool,
    distance_traveled: float,
    min_distance_threshold: float = 0.5,
) -> bool:
    """
    Evaluates whether an episode counts as 'survived'.

    Definition:
        A rollout is considered survived if and only if:
        1. No torso/body ground contact occurred (robot did not fall).
        2. Total forward distance traveled meets or exceeds min_distance_threshold.

    Args:
        has_torso_contact: True if torso/base hit ground during the episode.
        distance_traveled: Total forward distance traveled (m).
        min_distance_threshold: Minimum required forward progress (m). Default 0.5m.

    Returns:
        bool: True if survived, False otherwise.
    """
    if has_torso_contact:
        return False
    if distance_traveled < min_distance_threshold:
        return False
    return True
