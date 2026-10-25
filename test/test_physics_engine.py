"""
Unit tests for Quadruped Multi-Body Dynamics Physics Engine.

Author: Youseef Abdel Tawwab
"""

import numpy as np
import pytest
from sim2real_rl.envs.physics_engine import QuadrupedPhysicsEngine, PhysicsConfig
from sim2real_rl.envs.terrain import ProceduralTerrain, TerrainConfig, TerrainType


def test_physics_engine_initialization():
    """Verify physics engine initializes state vectors and nominal joint angles correctly."""
    engine = QuadrupedPhysicsEngine()
    engine.reset()

    assert engine.base_pos[2] == 0.28
    np.testing.assert_allclose(engine.base_quat, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert len(engine.joint_pos) == 12
    assert len(engine.joint_vel) == 12
    assert engine.foot_positions_world.shape == (4, 3)


def test_foot_kinematics_consistency():
    """Verify all 4 foot positions are below the base and have symmetric lateral offsets."""
    engine = QuadrupedPhysicsEngine()
    engine.reset()

    # Base at z = 0.28
    for i in range(4):
        foot_z = engine.foot_positions_world[i, 2]
        assert foot_z < engine.base_pos[2], f"Foot {i} must be below the base"

    # FL (leg 0) and RL (leg 2) are on the left (positive y)
    # FR (leg 1) and RR (leg 3) are on the right (negative y)
    assert engine.foot_positions_world[0, 1] > 0.0  # FL
    assert engine.foot_positions_world[1, 1] < 0.0  # FR
    assert engine.foot_positions_world[2, 1] > 0.0  # RL
    assert engine.foot_positions_world[3, 1] < 0.0  # RR


def test_pd_motor_torque_limits():
    """Verify applied motor torques never exceed maximum rated torque."""
    cfg = PhysicsConfig(max_torque=35.5)
    engine = QuadrupedPhysicsEngine(config=cfg)
    engine.reset()

    # Request extreme joint targets far outside limits
    extreme_targets = np.full(12, 10.0)
    engine.step(extreme_targets)

    assert np.all(np.abs(engine.joint_torques) <= 35.5 + 1e-6)


def test_contact_forces_and_gravity():
    """Verify that when feet penetrate ground, positive normal contact forces are generated."""
    engine = QuadrupedPhysicsEngine()
    # Place robot slightly low so feet penetrate ground
    engine.reset(base_pos=np.array([0.0, 0.0, 0.25]))

    # Advance 1 step
    engine.step(engine.nominal_joint_angles)

    # At least some feet should be in contact and generating upward normal forces
    assert np.any(engine.foot_contacts)
    total_fz = np.sum(engine.contact_forces[:, 2])
    assert total_fz > 0.0


def test_external_perturbation():
    """Verify external perturbation force creates corresponding linear acceleration."""
    engine = QuadrupedPhysicsEngine()
    engine.reset()

    # Apply 20 N push in positive x direction
    engine.apply_perturbation(force=np.array([20.0, 0.0, 0.0]))
    engine.step(engine.nominal_joint_angles)

    # Base should acquire positive forward velocity
    assert engine.base_lin_vel[0] > 0.0
