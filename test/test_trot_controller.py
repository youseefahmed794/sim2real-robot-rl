"""
Unit tests for Analytical IK Trotting Gait Controller.

Author: Youseef Abdel Tawwab
"""

import numpy as np
import pytest

from sim2real_rl.baselines.trot_controller import AnalyticalTrotController
from sim2real_rl.utils.spatial_math import forward_kinematics_leg


def test_trot_controller_ik_fk_consistency():
    """Verify that leg IK precisely recovers target foot coordinates."""
    ctrl = AnalyticalTrotController()

    test_targets = [
        np.array([0.0, 0.08, -0.32]),
        np.array([0.05, 0.08, -0.30]),
        np.array([-0.05, 0.08, -0.33]),
    ]

    for p_des in test_targets:
        q_sol = ctrl.compute_leg_ik(p_des, is_left=True)
        p_recovered = forward_kinematics_leg(
            q_sol, is_left=True, l_hip=ctrl.l_hip, l_thigh=ctrl.l_thigh, l_calf=ctrl.l_calf
        )
        assert np.allclose(p_des, p_recovered, atol=1e-3), \
            f"IK/FK mismatch: Target={p_des}, Recovered={p_recovered}"


def test_trot_controller_actions_bounded():
    """Verify action outputs are within [-1, 1] bounds over multiple gait cycles."""
    ctrl = AnalyticalTrotController(stride_frequency=2.0)
    cmd_vel = np.array([0.4, 0.0, 0.0])
    dt = 0.02

    for _ in range(100):
        action = ctrl.get_action(dt=dt, cmd_vel=cmd_vel)
        assert action.shape == (12,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
        assert not np.isnan(action).any()
