"""
Spatial Math and Kinematics Utilities for Quadruped Robots.
Implements SO(3) rotations, quaternion operations, forward leg kinematics,
and leg Jacobian derivation for 3-DOF legged mechanisms.

Author: Youseef Abdel Tawwab
"""

import numpy as np
from typing import Tuple, List


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert ZYX Euler angles (roll, pitch, yaw) in radians to unit quaternion [w, x, y, z].
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    q = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_to_euler(quat: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert unit quaternion [w, x, y, z] to ZYX Euler angles (roll, pitch, yaw) in radians.
    """
    w, x, y, z = quat
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(roll), float(pitch), float(yaw)


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix R in SO(3).
    """
    w, x, y, z = quat
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)

    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)

    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    return np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]], dtype=np.float64)


def rot_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix R in SO(3) to unit quaternion [w, x, y, z].
    """
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def project_gravity_to_base(quat: np.ndarray, gravity_mag: float = 9.81) -> np.ndarray:
    """
    Project world vertical gravity vector [0, 0, -g] into the robot base frame:
    g_base = R_base^T * [0, 0, -g]^T
    """
    R = quat_to_rot_matrix(quat)
    g_world = np.array([0.0, 0.0, -gravity_mag], dtype=np.float64)
    return R.T @ g_world


def transform_vector_to_base(vec_world: np.ndarray, quat_base: np.ndarray) -> np.ndarray:
    """
    Transform a 3D vector from world frame to robot base frame.
    """
    R = quat_to_rot_matrix(quat_base)
    return R.T @ np.asarray(vec_world, dtype=np.float64)


def forward_kinematics_leg(
    q: np.ndarray,
    is_left: bool = True,
    l_hip: float = 0.08,
    l_thigh: float = 0.21,
    l_calf: float = 0.21
) -> np.ndarray:
    """
    Forward Kinematics of a 3-DOF Quadruped Leg (Hip Abduction/Adduction, Hip Pitch, Knee Pitch).
    Returns foot tip position [x, y, z] relative to the hip base origin.

    Args:
        q: [q_hip_roll, q_hip_pitch, q_knee_pitch] in radians
        is_left: True if left leg (positive y offset), False if right (negative y offset)
        l_hip: length of hip link (lateral offset)
        l_thigh: length of thigh link
        l_calf: length of calf link
    """
    q1, q2, q3 = q
    y_sign = 1.0 if is_left else -1.0

    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)

    # Leg position relative to hip roll axis
    x = -l_thigh * s2 - l_calf * s23
    y = y_sign * l_hip * c1 + l_thigh * s1 * c2 + l_calf * s1 * c23
    z = y_sign * l_hip * s1 - l_thigh * c1 * c2 - l_calf * c1 * c23

    return np.array([x, y, z], dtype=np.float64)


def leg_jacobian(
    q: np.ndarray,
    is_left: bool = True,
    l_hip: float = 0.08,
    l_thigh: float = 0.21,
    l_calf: float = 0.21
) -> np.ndarray:
    """
    Analytical Jacobian matrix J_leg(q) = d(p_foot)/d(q) in R^{3x3}.
    Relates joint velocities dq to foot linear velocity v_foot in hip frame:
    v_foot = J_leg * dq
    """
    q1, q2, q3 = q
    y_sign = 1.0 if is_left else -1.0

    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)

    J = np.zeros((3, 3), dtype=np.float64)

    # dx / dq
    J[0, 0] = 0.0
    J[0, 1] = -l_thigh * c2 - l_calf * c23
    J[0, 2] = -l_calf * c23

    # dy / dq
    J[1, 0] = -y_sign * l_hip * s1 + l_thigh * c1 * c2 + l_calf * c1 * c23
    J[1, 1] = -l_thigh * s1 * s2 - l_calf * s1 * s23
    J[1, 2] = -l_calf * s1 * s23

    # dz / dq
    J[2, 0] = y_sign * l_hip * c1 + l_thigh * s1 * c2 + l_calf * s1 * c23
    J[2, 1] = l_thigh * c1 * s2 + l_calf * c1 * s23
    J[2, 2] = l_calf * c1 * s23

    return J
