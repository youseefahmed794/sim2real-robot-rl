"""
Unit tests for SO(3) spatial math, quaternion algebra,
forward kinematics, and leg Jacobian computation.

Author: Youseef Abdel Tawwab
"""

import numpy as np
import pytest

from sim2real_rl.utils.spatial_math import (
    euler_to_quat,
    quat_to_euler,
    quat_to_rot_matrix,
    rot_matrix_to_quat,
    project_gravity_to_base,
    transform_vector_to_base,
    forward_kinematics_leg,
    leg_jacobian,
)


def test_euler_quat_roundtrip():
    """Verify Euler <-> Quaternion roundtrip conversion accuracy."""
    test_cases = [
        (0.0, 0.0, 0.0),
        (0.15, -0.25, 0.80),
        (-0.50, 0.30, -1.20),
        (0.0, np.pi / 4, 0.0),
    ]

    for roll, pitch, yaw in test_cases:
        q = euler_to_quat(roll, pitch, yaw)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-7)

        r_rec, p_rec, y_rec = quat_to_euler(q)
        assert np.isclose(roll, r_rec, atol=1e-5)
        assert np.isclose(pitch, p_rec, atol=1e-5)
        assert np.isclose(yaw, y_rec, atol=1e-5)


def test_so3_rotation_matrix_properties():
    """Verify that generated rotation matrices satisfy SO(3) orthogonality and unit determinant."""
    q = euler_to_quat(0.2, -0.4, 0.7)
    R = quat_to_rot_matrix(q)

    # R * R^T = I
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-7)
    # det(R) = +1
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-7)

    # Roundtrip R -> quat -> R
    q_rec = rot_matrix_to_quat(R)
    R_rec = quat_to_rot_matrix(q_rec)
    assert np.allclose(R, R_rec, atol=1e-7)


def test_project_gravity_to_base():
    """Verify gravity vector projection from world to robot base frame."""
    # When robot is perfectly level: g_base should be [0, 0, -9.81]
    q_level = euler_to_quat(0.0, 0.0, 0.0)
    g_base = project_gravity_to_base(q_level)
    assert np.allclose(g_base, [0.0, 0.0, -9.81], atol=1e-5)

    # When robot pitched 90 deg up: base x-axis points up, so g_base should point backward [-9.81, 0, 0]
    q_pitch90 = euler_to_quat(0.0, np.pi / 2.0, 0.0)
    g_pitch = project_gravity_to_base(q_pitch90)
    assert np.isclose(g_pitch[0], 9.81, atol=1e-4)
    assert np.isclose(g_pitch[2], 0.0, atol=1e-4)


def test_forward_kinematics_leg():
    """Verify 3-DOF leg forward kinematics for both left and right legs."""
    # Zero configuration
    q_zero = np.array([0.0, 0.0, 0.0])
    p_left = forward_kinematics_leg(q_zero, is_left=True, l_hip=0.08, l_thigh=0.21, l_calf=0.21)
    # At zero angles: x=0, y=l_hip (+0.08 for left), z=-(l_thigh + l_calf) = -0.42
    assert np.isclose(p_left[0], 0.0, atol=1e-6)
    assert np.isclose(p_left[1], 0.08, atol=1e-6)
    assert np.isclose(p_left[2], -0.42, atol=1e-6)

    # Right leg should have negative y
    p_right = forward_kinematics_leg(q_zero, is_left=False, l_hip=0.08, l_thigh=0.21, l_calf=0.21)
    assert np.isclose(p_right[1], -0.08, atol=1e-6)
    assert np.isclose(p_right[2], -0.42, atol=1e-6)


def test_leg_jacobian_finite_difference():
    """Verify analytical leg Jacobian against numerical finite differences."""
    q = np.array([0.15, 0.65, -1.30])
    is_left = True
    l_hip = 0.08
    l_thigh = 0.21
    l_calf = 0.21

    J_analytic = leg_jacobian(q, is_left, l_hip, l_thigh, l_calf)

    # Compute numerical Jacobian via central difference
    eps = 1e-6
    J_numeric = np.zeros((3, 3))
    for i in range(3):
        q_plus = q.copy()
        q_plus[i] += eps
        q_minus = q.copy()
        q_minus[i] -= eps

        p_plus = forward_kinematics_leg(q_plus, is_left, l_hip, l_thigh, l_calf)
        p_minus = forward_kinematics_leg(q_minus, is_left, l_hip, l_thigh, l_calf)

        J_numeric[:, i] = (p_plus - p_minus) / (2.0 * eps)

    assert np.allclose(J_analytic, J_numeric, atol=1e-5), \
        f"Jacobian mismatch:\nAnalytic:\n{J_analytic}\nNumeric:\n{J_numeric}"
