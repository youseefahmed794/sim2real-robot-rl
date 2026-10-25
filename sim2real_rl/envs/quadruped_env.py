"""
Gymnasium Locomotion Environment for 12-DOF Quadruped Robot.
Implements the 48-dimensional observation space, 12-dimensional continuous action space,
stochastic command resampling, multi-objective reward decomposition, and episode life-cycle.

Author: Youseef Abdel Tawwab
"""

import os
from typing import Dict, Tuple, Optional, Any
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Minimal fallback providing equivalent Gymnasium interface
    class gym:
        class Env:
            pass
    class spaces:
        class Box:
            def __init__(self, low, high, shape, dtype=np.float32):
                self.low = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.asarray(low, dtype=dtype)
                self.high = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.asarray(high, dtype=dtype)
                self.shape = shape
                self.dtype = dtype
            def sample(self):
                return np.random.uniform(self.low, self.high).astype(self.dtype)
            def contains(self, x):
                return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

from sim2real_rl.utils.spatial_math import (
    euler_to_quat,
    quat_to_euler,
    project_gravity_to_base,
    transform_vector_to_base,
)
from sim2real_rl.envs.terrain import ProceduralTerrain, TerrainConfig, TerrainType
from sim2real_rl.envs.rewards import RewardManager, RewardConfig
from sim2real_rl.envs.physics_engine import QuadrupedPhysicsEngine, PhysicsConfig


class QuadrupedEnv(gym.Env):
    """
    Gymnasium-compliant quadruped locomotion environment.
    Observation Space: R^48
        - Base linear velocity in body frame [3]
        - Base angular velocity in body frame [3]
        - Projected gravity vector in body frame [3]
        - Velocity commands [vx_cmd, vy_cmd, wz_cmd] [3]
        - Joint positions offset from nominal [12]
        - Joint velocities [12]
        - Previous actions [12]
    Action Space: R^12
        - Desired target joint position offsets in [-1.0, 1.0] scaled by action_scale.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        physics_config: Optional[PhysicsConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        terrain_config: Optional[TerrainConfig] = None,
        action_scale: float = 0.25,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.action_scale = action_scale
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Initialize modular subsystems
        self.terrain = ProceduralTerrain(terrain_config)
        self.physics = QuadrupedPhysicsEngine(physics_config, self.terrain)
        self.reward_manager = RewardManager(reward_config)

        # Joint limit arrays (12)
        # FL, FR, RL, RR: hip_roll, hip_pitch, knee_pitch
        self.joint_limits_lower = np.array([
            -0.80, -1.00, -2.70,
            -0.80, -1.00, -2.70,
            -0.80, -1.00, -2.70,
            -0.80, -1.00, -2.70
        ], dtype=np.float64)
        self.joint_limits_upper = np.array([
            0.80, 3.50, -0.60,
            0.80, 3.50, -0.60,
            0.80, 3.50, -0.60,
            0.80, 3.50, -0.60
        ], dtype=np.float64)

        # Velocity command ranges
        self.cmd_vx_range = (-0.5, 1.2)
        self.cmd_vy_range = (-0.4, 0.4)
        self.cmd_wz_range = (-1.0, 1.0)
        self.cmd_vel = np.zeros(3, dtype=np.float64)

        # Define Observation & Action Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32
        )

        # Episode state variables
        self.step_count = 0
        self.current_action = np.zeros(12, dtype=np.float64)
        self.prev_action = np.zeros(12, dtype=np.float64)
        self._np_random = np.random.RandomState(42)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state with randomized commands and slight pose perturbations.
        """
        if seed is not None:
            self._np_random = np.random.RandomState(seed)

        self.step_count = 0
        self.current_action = np.zeros(12, dtype=np.float64)
        self.prev_action = np.zeros(12, dtype=np.float64)
        self.reward_manager.reset()

        # Sample velocity commands
        if options is not None and "cmd_vel" in options:
            self.cmd_vel = np.asarray(options["cmd_vel"], dtype=np.float64)
        else:
            self.cmd_vel = np.array([
                self._np_random.uniform(*self.cmd_vx_range),
                self._np_random.uniform(*self.cmd_vy_range),
                self._np_random.uniform(*self.cmd_wz_range),
            ], dtype=np.float64)

        # Reset robot with slight stochastic perturbations
        init_pos = np.array([
            0.0,
            0.0,
            0.28 + self._np_random.uniform(-0.01, 0.01)
        ], dtype=np.float64)

        init_roll = self._np_random.uniform(-0.03, 0.03)
        init_pitch = self._np_random.uniform(-0.03, 0.03)
        init_yaw = self._np_random.uniform(-np.pi, np.pi)
        init_quat = euler_to_quat(init_roll, init_pitch, init_yaw)

        init_joints = self.physics.nominal_joint_angles + self._np_random.uniform(-0.05, 0.05, size=12)
        init_joints = np.clip(init_joints, self.joint_limits_lower, self.joint_limits_upper)

        self.physics.reset(
            base_pos=init_pos,
            base_quat=init_quat,
            joint_pos=init_joints,
            joint_vel=np.zeros(12, dtype=np.float64)
        )

        obs = self._get_observation()
        info = {
            "cmd_vel": self.cmd_vel.copy(),
            "base_pos": self.physics.base_pos.copy(),
            "step_count": 0
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        """
        Construct the 48-dimensional observation vector s_t:
        [base_lin_vel_b (3), base_ang_vel_b (3), proj_gravity (3), cmd_vel (3),
         joint_pos_err (12), joint_vel (12), prev_action (12)]
        """
        base_lin_vel_b = self.physics.get_base_lin_vel_body()
        base_ang_vel_b = self.physics.base_ang_vel
        proj_gravity = self.physics.get_projected_gravity() / 9.81  # Normalized to unit magnitude
        joint_pos_err = self.physics.joint_pos - self.physics.nominal_joint_angles
        joint_vel = self.physics.joint_vel
        prev_act = self.current_action

        obs = np.concatenate([
            base_lin_vel_b,          # 3
            base_ang_vel_b,          # 3
            proj_gravity,            # 3
            self.cmd_vel,            # 3
            joint_pos_err,           # 12
            joint_vel,               # 12
            prev_act                 # 12
        ]).astype(np.float32)

        return obs

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply RL action, advance physics, evaluate reward and episode termination.
        """
        self.step_count += 1
        clipped_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.prev_action = np.copy(self.current_action)
        self.current_action = np.copy(clipped_action)

        # Target joint positions: q_des = q_nominal + k_scale * action
        q_target = self.physics.nominal_joint_angles + self.action_scale * clipped_action
        q_target = np.clip(q_target, self.joint_limits_lower, self.joint_limits_upper)

        # Advance physics simulation through decimation substeps
        self.physics.step(q_target)

        # Extract post-step states
        base_lin_vel_b = self.physics.get_base_lin_vel_body()
        base_ang_vel_b = self.physics.base_ang_vel
        proj_gravity = self.physics.get_projected_gravity()
        roll, pitch, _ = quat_to_euler(self.physics.base_quat)

        # Check termination (tip over or structural collapse)
        terminated = False
        if abs(roll) > 0.85 or abs(pitch) > 0.85:
            # Robot tipped over (~48 degrees)
            terminated = True
        elif self.physics.base_pos[2] < 0.12:
            # Robot fell/collapsed
            terminated = True

        # Check truncation (max steps reached)
        truncated = self.step_count >= self.max_episode_steps

        # Compute multi-objective reward
        control_dt = self.physics.config.substep_dt * self.physics.config.decimation
        reward, breakdown = self.reward_manager.compute_reward(
            base_lin_vel_base=base_lin_vel_b,
            base_ang_vel_base=base_ang_vel_b,
            projected_gravity=proj_gravity,
            cmd_vel=self.cmd_vel,
            base_pos_z=float(self.physics.base_pos[2]),
            joint_pos=self.physics.joint_pos,
            joint_vel=self.physics.joint_vel,
            joint_torques=self.physics.joint_torques,
            current_action=self.current_action,
            foot_contacts=self.physics.foot_contacts,
            foot_velocities=self.physics.foot_velocities_world,
            joint_limits_lower=self.joint_limits_lower,
            joint_limits_upper=self.joint_limits_upper,
            dt=control_dt
        )

        # Penalize hard termination
        if terminated:
            reward -= 5.0

        obs = self._get_observation()
        info = {
            "reward_breakdown": breakdown,
            "base_pos": self.physics.base_pos.copy(),
            "base_lin_vel": self.physics.base_lin_vel.copy(),
            "foot_contacts": self.physics.foot_contacts.copy(),
            "cmd_vel": self.cmd_vel.copy(),
            "step_count": self.step_count
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Render stub for Gymnasium interface."""
        return None

    def close(self):
        """Cleanup environment resources."""
        pass
