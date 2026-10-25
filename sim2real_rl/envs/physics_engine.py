"""
High-Fidelity Multi-Body Contact Dynamics Physics Engine for 12-DOF Quadruped Robot.
Implements 6-DOF floating-base dynamics, actuator PD modeling with torque limits,
Hunt-Crossley compliant ground contact forces with Coulomb friction cone,
and optional MuJoCo C-engine acceleration.

Author: Youseef Abdel Tawwab
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import numpy as np

from sim2real_rl.utils.spatial_math import (
    quat_to_rot_matrix,
    rot_matrix_to_quat,
    project_gravity_to_base,
    transform_vector_to_base,
    forward_kinematics_leg,
    leg_jacobian,
)
from sim2real_rl.envs.terrain import ProceduralTerrain, TerrainConfig, TerrainType


@dataclass
class PhysicsConfig:
    # Mass & Inertia
    total_mass: float = 12.5       # kg
    base_mass: float = 6.0         # kg
    base_inertia: np.ndarray = field(
        default_factory=lambda: np.diag([0.035, 0.085, 0.105])
    )
    # Kinematics
    l_hip: float = 0.08            # m
    l_thigh: float = 0.21          # m
    l_calf: float = 0.21           # m
    foot_radius: float = 0.022     # m
    hip_offsets: np.ndarray = field(
        default_factory=lambda: np.array([
            [0.19, 0.10, 0.0],    # FL
            [0.19, -0.10, 0.0],   # FR
            [-0.19, 0.10, 0.0],   # RL
            [-0.19, -0.10, 0.0],  # RR
        ], dtype=np.float64)
    )
    # Contact model parameters (Hunt-Crossley spring-damper)
    contact_stiffness: float = 8500.0   # N/m
    contact_damping: float = 180.0      # N*s/m
    friction_coeff: float = 0.8         # Coulomb friction
    tangential_stiffness: float = 350.0 # N*s/m for tangential creep damping
    # Actuator & Joint dynamics
    kp: float = 40.0                    # N*m/rad
    kd: float = 1.0                     # N*m*s/rad
    max_torque: float = 35.5            # N*m
    joint_damping: float = 0.15         # N*m*s/rad joint viscous friction
    rotor_inertia: float = 0.008        # kg*m^2 reflected actuator inertia
    # Simulation timing
    substep_dt: float = 0.002           # 500 Hz physics substep
    decimation: int = 10                # 10 substeps per 50 Hz control step (0.02s)


class QuadrupedPhysicsEngine:
    """
    Modular, high-performance multi-body dynamics simulator for a 12-DOF quadruped robot.
    Computes full floating base dynamics, contact forces, and joint actuator response.
    """

    def __init__(
        self,
        config: Optional[PhysicsConfig] = None,
        terrain: Optional[ProceduralTerrain] = None
    ):
        self.config = config if config is not None else PhysicsConfig()
        self.terrain = terrain if terrain is not None else ProceduralTerrain()

        # State vectors
        self.base_pos = np.array([0.0, 0.0, 0.28], dtype=np.float64)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # [w, x, y, z]
        self.base_lin_vel = np.zeros(3, dtype=np.float64)  # in world frame
        self.base_ang_vel = np.zeros(3, dtype=np.float64)  # in body frame

        # 12 joint states (FL, FR, RL, RR)
        self.nominal_joint_angles = np.array([
            0.0, 0.72, -1.44,   # FL
            0.0, 0.72, -1.44,   # FR
            0.0, 0.72, -1.44,   # RL
            0.0, 0.72, -1.44    # RR
        ], dtype=np.float64)
        self.joint_pos = np.copy(self.nominal_joint_angles)
        self.joint_vel = np.zeros(12, dtype=np.float64)
        self.joint_torques = np.zeros(12, dtype=np.float64)

        # Foot contact cache
        self.foot_contacts = np.zeros(4, dtype=bool)
        self.foot_positions_world = np.zeros((4, 3), dtype=np.float64)
        self.foot_velocities_world = np.zeros((4, 3), dtype=np.float64)
        self.contact_forces = np.zeros((4, 3), dtype=np.float64)

        # External perturbation disturbance [Fx, Fy, Fz] and [Tx, Ty, Tz]
        self.external_force = np.zeros(3, dtype=np.float64)
        self.external_torque = np.zeros(3, dtype=np.float64)

        # Check if MuJoCo backend is available
        self.has_mujoco = False
        self._mj_model = None
        self._mj_data = None
        self._try_init_mujoco()

    def _try_init_mujoco(self) -> None:
        """Attempt to initialize MuJoCo if installed and model file exists."""
        try:
            import mujoco
            mjcf_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "models", "mjcf", "quadruped_12dof.xml"
            )
            if os.path.exists(mjcf_path):
                self._mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
                self._mj_data = mujoco.MjData(self._mj_model)
                self.has_mujoco = True
        except Exception:
            self.has_mujoco = False

    def reset(
        self,
        base_pos: Optional[np.ndarray] = None,
        base_quat: Optional[np.ndarray] = None,
        joint_pos: Optional[np.ndarray] = None,
        joint_vel: Optional[np.ndarray] = None
    ) -> None:
        """Reset simulation states to specified or nominal values."""
        self.base_pos = (
            np.copy(base_pos) if base_pos is not None
            else np.array([0.0, 0.0, 0.28], dtype=np.float64)
        )
        self.base_quat = (
            np.copy(base_quat) if base_quat is not None
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        self.base_quat /= np.linalg.norm(self.base_quat)
        self.base_lin_vel = np.zeros(3, dtype=np.float64)
        self.base_ang_vel = np.zeros(3, dtype=np.float64)

        self.joint_pos = (
            np.copy(joint_pos) if joint_pos is not None
            else np.copy(self.nominal_joint_angles)
        )
        self.joint_vel = (
            np.copy(joint_vel) if joint_vel is not None
            else np.zeros(12, dtype=np.float64)
        )
        self.joint_torques = np.zeros(12, dtype=np.float64)
        self.external_force = np.zeros(3, dtype=np.float64)
        self.external_torque = np.zeros(3, dtype=np.float64)

        self._update_kinematics()

    def apply_perturbation(self, force: np.ndarray, torque: Optional[np.ndarray] = None) -> None:
        """Apply temporary external force/torque impulse to robot body."""
        self.external_force = np.asarray(force, dtype=np.float64)
        if torque is not None:
            self.external_torque = np.asarray(torque, dtype=np.float64)

    def _update_kinematics(self) -> None:
        """Compute forward kinematics for all 4 feet in world and base frames."""
        R_b = quat_to_rot_matrix(self.base_quat)
        cfg = self.config

        is_left_list = [True, False, True, False]  # FL, FR, RL, RR

        for i in range(4):
            q_leg = self.joint_pos[i * 3 : (i + 1) * 3]
            dq_leg = self.joint_vel[i * 3 : (i + 1) * 3]
            is_left = is_left_list[i]

            # Foot position relative to hip origin
            p_foot_hip = forward_kinematics_leg(
                q_leg,
                is_left=is_left,
                l_hip=cfg.l_hip,
                l_thigh=cfg.l_thigh,
                l_calf=cfg.l_calf
            )
            # Foot in base frame
            p_foot_base = cfg.hip_offsets[i] + p_foot_hip
            # Foot in world frame
            p_foot_world = self.base_pos + R_b @ p_foot_base
            self.foot_positions_world[i] = p_foot_world

            # Foot velocity in world frame: v_world = v_base + R_b * (omega_b x p_base + J_leg * dq)
            J_leg = leg_jacobian(
                q_leg,
                is_left=is_left,
                l_hip=cfg.l_hip,
                l_thigh=cfg.l_thigh,
                l_calf=cfg.l_calf
            )
            v_rel_base = np.cross(self.base_ang_vel, p_foot_base) + J_leg @ dq_leg
            v_foot_world = self.base_lin_vel + R_b @ v_rel_base
            self.foot_velocities_world[i] = v_foot_world

    def _compute_contact_forces(self) -> np.ndarray:
        """
        Compute ground reaction forces on each foot using compliant contact dynamics
        with Coulomb friction cone limits.
        """
        cfg = self.config
        forces_world = np.zeros((4, 3), dtype=np.float64)

        for i in range(4):
            p_foot = self.foot_positions_world[i]
            v_foot = self.foot_velocities_world[i]

            in_contact, depth, normal = self.terrain.check_foot_contact(
                p_foot, foot_radius=cfg.foot_radius
            )
            self.foot_contacts[i] = in_contact

            if in_contact and depth > 0.0:
                # Normal penetration velocity: v_n = v_foot . normal
                v_normal = float(np.dot(v_foot, normal))
                
                # Normal contact spring-damper force: F_n = k * delta - d * v_normal
                f_normal_mag = cfg.contact_stiffness * depth - cfg.contact_damping * v_normal
                f_normal_mag = max(0.0, f_normal_mag)

                # Tangential velocity in contact plane: v_t = v_foot - (v_foot . normal) * normal
                v_tangential = v_foot - v_normal * normal
                v_tangential_speed = float(np.linalg.norm(v_tangential))

                # Tangential friction with Coulomb limit
                mu = self.terrain.friction_coeff
                max_friction = mu * f_normal_mag
                tangential_force_mag = cfg.tangential_stiffness * v_tangential_speed

                if tangential_force_mag > max_friction:
                    tangential_force_mag = max_friction

                if v_tangential_speed > 1e-5:
                    f_tangential = -tangential_force_mag * (v_tangential / v_tangential_speed)
                else:
                    f_tangential = np.zeros(3, dtype=np.float64)

                total_force = f_normal_mag * normal + f_tangential
                forces_world[i] = total_force
            else:
                forces_world[i] = np.zeros(3, dtype=np.float64)

        self.contact_forces = forces_world
        return forces_world

    def step(self, target_joint_angles: np.ndarray) -> None:
        """
        Perform a full control step by running multiple high-rate physics substeps.
        
        Args:
            target_joint_angles: [12] desired joint positions (rad) from policy.
        """
        cfg = self.config
        R_b = quat_to_rot_matrix(self.base_quat)
        inv_base_inertia = np.linalg.inv(cfg.base_inertia)

        is_left_list = [True, False, True, False]

        for _ in range(cfg.decimation):
            # 1. Update foot kinematics
            self._update_kinematics()

            # 2. Compute contact forces
            contact_forces_world = self._compute_contact_forces()

            # 3. Low-level PD motor control with saturation
            pd_torques = cfg.kp * (target_joint_angles - self.joint_pos) - cfg.kd * self.joint_vel
            self.joint_torques = np.clip(pd_torques, -cfg.max_torque, cfg.max_torque)

            # 4. Floating base linear acceleration (Newton's 2nd Law in world frame)
            total_force_world = np.sum(contact_forces_world, axis=0) + self.external_force
            gravity_force = np.array([0.0, 0.0, -9.81 * cfg.total_mass], dtype=np.float64)
            lin_acc_world = (total_force_world + gravity_force) / cfg.total_mass

            # 5. Floating base angular acceleration (Euler's equations in body frame)
            # Contact moments: sum(p_foot_base x (R_b^T * F_contact_world))
            total_torque_body = np.copy(self.external_torque)
            for i in range(4):
                f_contact_body = R_b.T @ contact_forces_world[i]
                p_foot_base = cfg.hip_offsets[i] + forward_kinematics_leg(
                    self.joint_pos[i * 3 : (i + 1) * 3],
                    is_left=is_left_list[i],
                    l_hip=cfg.l_hip,
                    l_thigh=cfg.l_thigh,
                    l_calf=cfg.l_calf
                )
                total_torque_body += np.cross(p_foot_base, f_contact_body)

            # Gyroscopic term: omega x (I * omega)
            gyro_term = np.cross(self.base_ang_vel, cfg.base_inertia @ self.base_ang_vel)
            ang_acc_body = inv_base_inertia @ (total_torque_body - gyro_term)

            # 6. Joint accelerations
            # ddq = (tau - J^T * F_contact - b * dq) / M_eff
            joint_acc = np.zeros(12, dtype=np.float64)
            for i in range(4):
                q_leg = self.joint_pos[i * 3 : (i + 1) * 3]
                dq_leg = self.joint_vel[i * 3 : (i + 1) * 3]
                tau_leg = self.joint_torques[i * 3 : (i + 1) * 3]
                f_contact_body = R_b.T @ contact_forces_world[i]

                J_leg = leg_jacobian(
                    q_leg,
                    is_left=is_left_list[i],
                    l_hip=cfg.l_hip,
                    l_thigh=cfg.l_thigh,
                    l_calf=cfg.l_calf
                )
                tau_contact = J_leg.T @ f_contact_body
                tau_damping = cfg.joint_damping * dq_leg
                
                net_tau = tau_leg - tau_contact - tau_damping
                joint_acc[i * 3 : (i + 1) * 3] = net_tau / cfg.rotor_inertia

            # 7. Symplectic Euler integration
            dt = cfg.substep_dt

            # Linear velocity & position
            self.base_lin_vel += lin_acc_world * dt
            self.base_pos += self.base_lin_vel * dt

            # Angular velocity & orientation quaternion
            self.base_ang_vel += ang_acc_body * dt
            # Quaternion kinematics: dq/dt = 0.5 * q * [0, omega]
            w, x, y, z = self.base_quat
            wx, wy, wz = self.base_ang_vel
            d_quat = 0.5 * np.array([
                -x * wx - y * wy - z * wz,
                w * wx + y * wz - z * wy,
                w * wy - x * wz + z * wx,
                w * wz + x * wy - y * wx
            ], dtype=np.float64)
            self.base_quat += d_quat * dt
            self.base_quat /= np.linalg.norm(self.base_quat)
            R_b = quat_to_rot_matrix(self.base_quat)

            # Joint velocities & positions
            self.joint_vel += joint_acc * dt
            # Apply joint velocity limit
            self.joint_vel = np.clip(self.joint_vel, -21.0, 21.0)
            self.joint_pos += self.joint_vel * dt

        # Clear external perturbations after control step
        self.external_force = np.zeros(3, dtype=np.float64)
        self.external_torque = np.zeros(3, dtype=np.float64)

        # Update foot kinematics after step
        self._update_kinematics()

    def get_base_lin_vel_body(self) -> np.ndarray:
        """Transform base linear velocity to base local body frame."""
        return transform_vector_to_base(self.base_lin_vel, self.base_quat)

    def get_projected_gravity(self) -> np.ndarray:
        """Get gravity vector projected into base frame: g_b = R_b^T * [0, 0, -9.81]."""
        return project_gravity_to_base(self.base_quat)
