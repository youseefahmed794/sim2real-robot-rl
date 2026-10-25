"""
Native MuJoCo Locomotion Environment for 12-DOF Quadruped.

Interacts directly with the C-accelerated MuJoCo physics engine via official Python bindings.
Solves full multi-body rigid dynamics, non-linear Coriolis forces, and complementarity
ground contact constraints.

Author: Youseef Abdel Tawwab
"""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import mujoco

from sim2real_rl.envs.rewards import RewardManager, RewardConfig
from sim2real_rl.utils.spatial_math import quat_to_rot_matrix, quat_to_euler

# Optional Gymnasium base class
try:
    import gymnasium as gym
    from gymnasium import spaces
    BaseEnv = gym.Env
    GYM_AVAILABLE = True
except ImportError:
    BaseEnv = object
    spaces = None
    GYM_AVAILABLE = False


class MuJoCoQuadrupedEnv(BaseEnv):
    """
    MuJoCo-backed Quadruped Locomotion Environment.
    Loads models/mjcf/quadruped_12dof.xml and provides a standard Gymnasium-style step/reset API.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        model_path: Optional[str] = None,
        reward_config: Optional[RewardConfig] = None,
        decimation: int = 4,
        action_scale: float = 0.25,
        kp: float = 55.0,
        kd: float = 1.8,
        nominal_height: float = 0.325,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
    ):
        if GYM_AVAILABLE and BaseEnv is not object:
            super().__init__()

        self.render_mode = render_mode
        if model_path is None:
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "models", "mjcf", "quadruped_12dof.xml"
            )
            model_path = os.path.abspath(default_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo model file not found: {model_path}")

        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.decimation = decimation
        self.action_scale = action_scale
        self.kp = kp
        self.kd = kd
        self.nominal_height = nominal_height
        self.max_episode_steps = max_episode_steps

        # 12-DOF nominal standing pose [hip_roll, hip_pitch, knee_pitch] x 4
        self.q_nominal = np.array([0.0, 0.72, -1.44] * 4, dtype=np.float64)

        # Foot geom IDs for ground contact tracking
        self.foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        ]

        # Space dimensions
        self.observation_dim = 48
        self.action_dim = 12

        if GYM_AVAILABLE and spaces is not None:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float64
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float64
            )

        # Reward engine
        self.reward_manager = RewardManager(config=reward_config)

        # State tracking
        self.step_count = 0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float64)
        self.cmd_vel = np.array([0.5, 0.0, 0.0], dtype=np.float64)  # [vx, vy, omega_z]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        cmd_vel: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset simulation to nominal standing posture with small initial noise."""
        if GYM_AVAILABLE and BaseEnv is not object:
            super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Reset base position [x, y, z] and orientation quaternion [w, x, y, z]
        self.data.qpos[0:3] = [0.0, 0.0, self.nominal_height]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # Initialize joint angles near nominal standing configuration
        q_init = self.q_nominal + np.random.uniform(-0.02, 0.02, size=self.action_dim)
        self.data.qpos[7:19] = q_init
        self.data.qvel[:] = 0.0

        # Step forward to settle contacts
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float64)
        if cmd_vel is not None:
            self.cmd_vel = np.asarray(cmd_vel, dtype=np.float64)
        elif options is not None and "cmd_vel" in options:
            self.cmd_vel = np.asarray(options["cmd_vel"], dtype=np.float64)
        else:
            self.cmd_vel = np.array([0.5, 0.0, 0.0], dtype=np.float64)

        obs = self._get_obs()
        info = {
            "cmd_vel": self.cmd_vel.copy(),
            "base_pos": self.data.qpos[0:3].copy(),
            "step_count": 0,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one control cycle with low-level PD actuator sub-stepping.
        """
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_target = self.q_nominal + action * self.action_scale

        applied_torques = []
        # Multi-step PD execution at physics frequency
        for _ in range(self.decimation):
            q_current = self.data.qpos[7:19]
            dq_current = self.data.qvel[6:18]
            tau = self.kp * (q_target - q_current) - self.kd * dq_current
            tau = np.clip(tau, -35.5, 35.5)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
            applied_torques.append(tau)

        mean_tau = np.mean(applied_torques, axis=0)
        self.step_count += 1

        obs = self._get_obs()

        # Extract state for reward and termination
        base_pos = self.data.qpos[0:3].copy()
        base_quat = self.data.qpos[3:7].copy()
        base_lin_vel_world = self.data.qvel[0:3].copy()
        base_ang_vel_base = self.data.qvel[3:6].copy()

        R_base = quat_to_rot_matrix(base_quat)
        base_lin_vel_base = R_base.T @ base_lin_vel_world

        # Foot contact force detection
        foot_contacts = self._get_foot_contacts()

        # Projected gravity in base frame
        projected_gravity = R_base.T @ np.array([0.0, 0.0, -1.0])

        # Joint limit bounds
        q_limits_lower = np.array([-0.80, -1.00, -2.70] * 4, dtype=np.float64)
        q_limits_upper = np.array([0.80, 3.50, -0.60] * 4, dtype=np.float64)

        # Foot linear velocities in world frame
        foot_velocities = np.zeros((4, 3), dtype=np.float64)

        # Compute rewards
        reward, reward_dict = self.reward_manager.compute_reward(
            base_lin_vel_base=base_lin_vel_base,
            base_ang_vel_base=base_ang_vel_base,
            projected_gravity=projected_gravity,
            cmd_vel=self.cmd_vel,
            base_pos_z=float(base_pos[2]),
            joint_pos=self.data.qpos[7:19],
            joint_vel=self.data.qvel[6:18],
            joint_torques=mean_tau,
            current_action=action,
            foot_contacts=foot_contacts,
            foot_velocities=foot_velocities,
            joint_limits_lower=q_limits_lower,
            joint_limits_upper=q_limits_upper,
            dt=self.model.opt.timestep * self.decimation,
        )

        self.prev_action = action.copy()

        # Termination checks
        roll, pitch, _ = quat_to_euler(base_quat)
        height = base_pos[2]

        terminated = bool(
            height < 0.18  # Body dropped
            or height > 0.55  # Excessive launch
            or abs(roll) > 0.75  # Tumbled roll
            or abs(pitch) > 0.75  # Tumbled pitch
        )

        truncated = bool(self.step_count >= self.max_episode_steps)

        info = {
            "reward_breakdown": reward_dict,
            "base_height": float(height),
            "base_roll": float(roll),
            "base_pitch": float(pitch),
            "forward_vel": float(base_lin_vel_base[0]),
            "mean_torque_norm": float(np.linalg.norm(mean_tau)),
            "foot_contacts": foot_contacts.tolist(),
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        Assemble 48-dimensional observation vector:
        - Base angular velocity (3)
        - Projected gravity in base frame (3)
        - Commanded velocity (3)
        - Joint positions deviation from nominal (12)
        - Joint velocities (12)
        - Previous actions (12)
        - Foot contact binary flags (3 + 1 padding = 4, or zero-padded to reach 48)
        """
        base_quat = self.data.qpos[3:7]
        R_base = quat_to_rot_matrix(base_quat)

        # Base angular velocity in base frame
        omega_base = self.data.qvel[3:6]

        # Projected gravity in base frame: R^T * [0, 0, -1]
        gravity_base = R_base.T @ np.array([0.0, 0.0, -1.0])

        # Joint states
        q_diff = self.data.qpos[7:19] - self.q_nominal
        dq = self.data.qvel[6:18]

        # Foot contacts (4)
        foot_contacts = self._get_foot_contacts().astype(np.float64)

        # 3 + 3 + 3 + 12 + 12 + 12 + 3 (padding) = 48
        obs = np.concatenate([
            omega_base,          # 3
            gravity_base,        # 3
            self.cmd_vel,        # 3
            q_diff,              # 12
            dq,                  # 12
            self.prev_action,    # 12
            foot_contacts[0:3],  # 3 (total = 48)
        ])

        return obs.astype(np.float64)

    def _get_foot_contacts(self) -> np.ndarray:
        """Query MuJoCo contact manifold to check which feet are touching the ground floor."""
        contacts = np.zeros(4, dtype=bool)
        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for foot_idx, fg_id in enumerate(self.foot_geom_ids):
                if (con.geom1 == fg_id and con.geom2 == floor_geom_id) or \
                   (con.geom2 == fg_id and con.geom1 == floor_geom_id):
                    contacts[foot_idx] = True

        return contacts
