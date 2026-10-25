"""
Unit tests for Multi-Objective Reward Formulation and Decomposition.

Author: Youseef Abdel Tawwab
"""

import numpy as np
import pytest
from sim2real_rl.envs.rewards import RewardManager, RewardConfig


def test_velocity_tracking_perfect():
    """Verify velocity tracking rewards achieve maximum value when error is zero."""
    mgr = RewardManager()
    mgr.reset()

    cmd_vel = np.array([0.8, 0.0, 0.5])
    base_lin_vel = np.array([0.8, 0.0, 0.0])  # Perfect horizontal match
    base_ang_vel = np.array([0.0, 0.0, 0.5])  # Perfect yaw rate match
    proj_gravity = np.array([0.0, 0.0, -9.81]) # Upright

    total_rew, breakdown = mgr.compute_reward(
        base_lin_vel_base=base_lin_vel,
        base_ang_vel_base=base_ang_vel,
        projected_gravity=proj_gravity,
        cmd_vel=cmd_vel,
        base_pos_z=0.28,
        joint_pos=np.zeros(12),
        joint_vel=np.zeros(12),
        joint_torques=np.zeros(12),
        current_action=np.zeros(12),
        foot_contacts=np.ones(4, dtype=bool),
        foot_velocities=np.zeros((4, 3)),
        joint_limits_lower=np.full(12, -2.0),
        joint_limits_upper=np.full(12, 2.0)
    )

    # w_lin_vel = 1.5, w_ang_vel = 0.8, alive = 1.0 => total ~ 3.3
    assert np.isclose(breakdown["rew_lin_vel"], 1.5, atol=1e-5)
    assert np.isclose(breakdown["rew_ang_vel"], 0.8, atol=1e-5)
    assert np.isclose(breakdown["rew_alive"], 1.0, atol=1e-5)
    assert np.isclose(breakdown["pen_base_height"], 0.0, atol=1e-5)
    assert total_rew > 3.0


def test_velocity_tracking_decay():
    """Verify velocity tracking reward decreases smoothly as error grows."""
    mgr = RewardManager()
    mgr.reset()

    cmd_vel = np.array([1.0, 0.0, 0.0])

    _, breakdown_close = mgr.compute_reward(
        base_lin_vel_base=np.array([0.9, 0.0, 0.0]),
        base_ang_vel_base=np.zeros(3),
        projected_gravity=np.array([0.0, 0.0, -9.81]),
        cmd_vel=cmd_vel,
        base_pos_z=0.28,
        joint_pos=np.zeros(12),
        joint_vel=np.zeros(12),
        joint_torques=np.zeros(12),
        current_action=np.zeros(12),
        foot_contacts=np.zeros(4, dtype=bool),
        foot_velocities=np.zeros((4, 3)),
        joint_limits_lower=np.full(12, -2.0),
        joint_limits_upper=np.full(12, 2.0)
    )

    _, breakdown_far = mgr.compute_reward(
        base_lin_vel_base=np.array([0.2, 0.0, 0.0]),
        base_ang_vel_base=np.zeros(3),
        projected_gravity=np.array([0.0, 0.0, -9.81]),
        cmd_vel=cmd_vel,
        base_pos_z=0.28,
        joint_pos=np.zeros(12),
        joint_vel=np.zeros(12),
        joint_torques=np.zeros(12),
        current_action=np.zeros(12),
        foot_contacts=np.zeros(4, dtype=bool),
        foot_velocities=np.zeros((4, 3)),
        joint_limits_lower=np.full(12, -2.0),
        joint_limits_upper=np.full(12, 2.0)
    )

    assert breakdown_close["rew_lin_vel"] > breakdown_far["rew_lin_vel"]


def test_penalties():
    """Verify individual penalties trigger and produce negative reward values."""
    mgr = RewardManager()
    mgr.reset()

    # Tilt body by 45 degrees: projected gravity has strong horizontal component
    proj_g_tilted = np.array([9.81 * np.sin(np.pi / 4), 0.0, -9.81 * np.cos(np.pi / 4)])

    _, breakdown = mgr.compute_reward(
        base_lin_vel_base=np.array([0.0, 0.0, 0.5]),  # vz != 0
        base_ang_vel_base=np.array([0.4, 0.3, 0.0]), # wx, wy != 0
        projected_gravity=proj_g_tilted,
        cmd_vel=np.zeros(3),
        base_pos_z=0.18,                              # height error
        joint_pos=np.zeros(12),
        joint_vel=np.full(12, 2.0),
        joint_torques=np.full(12, 10.0),             # high torque
        current_action=np.full(12, 0.5),             # jump from 0.0 action
        foot_contacts=np.ones(4, dtype=bool),
        foot_velocities=np.full((4, 3), 0.3),        # foot slip
        joint_limits_lower=np.full(12, -2.0),
        joint_limits_upper=np.full(12, 2.0)
    )

    assert breakdown["pen_lin_vel_z"] < 0.0
    assert breakdown["pen_ang_vel_xy"] < 0.0
    assert breakdown["pen_orientation"] < 0.0
    assert breakdown["pen_base_height"] < 0.0
    assert breakdown["pen_torques"] < 0.0
    assert breakdown["pen_mechanical_power"] < 0.0
    assert breakdown["pen_action_rate"] < 0.0
    assert breakdown["pen_foot_slip"] < 0.0
