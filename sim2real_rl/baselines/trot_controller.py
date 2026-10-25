"""
Analytical Inverse Kinematics Trotting Gait Controller (Classical Baseline).

Implements a diagonal trotting gait (FL-RR and FR-RL pairs) with:
- Cycloidal foot trajectory generator (swing clearance height and stance push)
- Analytical inverse kinematics for 3-DOF legs
- Phase-synchronized foot coordination

Serves as a rigorous classical control baseline for comparison against learned RL policies.

Author: Youseef Abdel Tawwab
"""

import math
from typing import Tuple, List, Optional
import numpy as np


class AnalyticalTrotController:
    """
    Classical Model-Based Trotting Gait Controller.
    """

    def __init__(
        self,
        stride_frequency: float = 1.5,     # Hz (tuned from 2.0 Hz for swing foot clearance)
        step_height: float = 0.08,         # m (tuned from 0.045 m to eliminate foot stubbing)
        nominal_height: float = 0.325,     # m (standing base height)
        l_hip: float = 0.08,               # m
        l_thigh: float = 0.21,             # m
        l_calf: float = 0.21,              # m
        action_scale: float = 0.25,        # scaling to match RL action bounds
    ):
        self.stride_frequency = stride_frequency
        self.stride_period = 1.0 / stride_frequency
        self.step_height = step_height
        self.nominal_height = nominal_height
        self.l_hip = l_hip
        self.l_thigh = l_thigh
        self.l_calf = l_calf
        self.action_scale = action_scale

        self.q_nominal = np.array([0.0, 0.72, -1.44] * 4, dtype=np.float64)
        self.time = 0.0

    def reset(self) -> None:
        """Reset internal gait clock."""
        self.time = 0.0

    def compute_leg_ik(
        self, target_pos: np.ndarray, is_left: bool = True
    ) -> np.ndarray:
        """
        Analytical 3-DOF Leg Inverse Kinematics.
        Maps target foot position [x, y, z] in hip frame to joint angles [q1, q2, q3].

        q1: Hip Abduction/Adduction (roll)
        q2: Hip Pitch (thigh)
        q3: Knee Pitch (calf)
        """
        x, y, z = target_pos
        y_sign = 1.0 if is_left else -1.0

        # Hip roll angle: projection in y-z plane
        # y = y_sign * l_hip * cos(q1) ...
        # Distance in y-z plane from roll axis to foot
        r_yz = math.sqrt(max(y**2 + z**2 - self.l_hip**2, 1e-8))
        q1 = math.atan2(y, -z) - math.atan2(y_sign * self.l_hip, r_yz)

        # In-plane distance from thigh joint to foot
        L_eff = math.sqrt(max(x**2 + r_yz**2, 1e-8))
        # Clamp to reachable workspace to avoid NaN
        L_clamped = min(L_eff, self.l_thigh + self.l_calf - 1e-4)

        # Law of cosines for knee angle q3
        cos_q3 = (L_clamped**2 - self.l_thigh**2 - self.l_calf**2) / (
            2.0 * self.l_thigh * self.l_calf
        )
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        # Knee bends backward: q3 in [-pi, 0]
        q3 = -math.acos(cos_q3)

        # Hip pitch angle q2
        phi1 = math.atan2(-x, r_yz)
        phi2 = math.acos(
            np.clip(
                (self.l_thigh**2 + L_clamped**2 - self.l_calf**2)
                / (2.0 * self.l_thigh * L_clamped),
                -1.0,
                1.0,
            )
        )
        q2 = phi1 + phi2

        return np.array([q1, q2, q3], dtype=np.float64)

    def get_action(
        self,
        dt: float,
        cmd_vel: np.ndarray,
        base_pitch: float = 0.0,
        base_roll: float = 0.0,
    ) -> np.ndarray:
        """
        Computes 12-dimensional normalized action vector a in [-1, 1] for current timestep.
        """
        self.time += dt
        phase = (self.time % self.stride_period) / self.stride_period

        # Diagonal leg pairs:
        # Pair 0 (FL & RR): in swing when phase in [0, 0.5)
        # Pair 1 (FR & RL): in swing when phase in [0.5, 1.0)
        vx = cmd_vel[0]
        step_length = vx * (self.stride_period * 0.5)

        q_target = np.zeros(12, dtype=np.float64)

        # Process each leg: FL (0), FR (1), RL (2), RR (3)
        legs_info = [
            (0, True, 0),    # FL: leg_idx=0, is_left=True, pair=0
            (1, False, 1),   # FR: leg_idx=1, is_left=False, pair=1
            (2, True, 1),    # RL: leg_idx=2, is_left=True, pair=1
            (3, False, 0),   # RR: leg_idx=3, is_left=False, pair=0
        ]

        for leg_idx, is_left, pair in legs_info:
            # Local phase for this leg pair: s in [0, 1]
            if pair == 0:
                is_swing = (phase < 0.5)
                s = phase / 0.5 if is_swing else (phase - 0.5) / 0.5
            else:
                is_swing = (phase >= 0.5)
                s = (phase - 0.5) / 0.5 if is_swing else phase / 0.5

            if is_swing:
                # Swing phase: foot lifts up and moves forward from -half to +half step
                x_foot = -0.5 * step_length + step_length * (s - 0.5 * math.sin(2.0 * math.pi * s) / math.pi)
                z_foot = -self.nominal_height + self.step_height * math.sin(math.pi * s)
            else:
                # Stance phase: foot on ground moving backward to push body forward
                x_foot = 0.5 * step_length - step_length * s
                z_foot = -self.nominal_height

            y_foot = self.l_hip if is_left else -self.l_hip
            target_pos = np.array([x_foot, y_foot, z_foot])

            q_leg = self.compute_leg_ik(target_pos, is_left=is_left)
            q_target[leg_idx * 3 : leg_idx * 3 + 3] = q_leg

        # Convert target joint angles to normalized action offsets: a = (q_target - q_nominal) / action_scale
        action = (q_target - self.q_nominal) / self.action_scale
        return np.clip(action, -1.0, 1.0)
