"""
Unit tests for CoT and evaluation metrics.

Author: Youseef Abdel Tawwab
"""

import math
import numpy as np
import pytest

from sim2real_rl.utils.metrics import (
    compute_mechanical_power,
    compute_cost_of_transport,
    evaluate_survival,
)


def test_compute_mechanical_power_known_positive():
    """Verify mechanical power calculation for known positive work."""
    torques = np.array([10.0, 5.0, 2.0])
    joint_vels = np.array([2.0, 3.0, 4.0])
    # 10*2 + 5*3 + 2*4 = 20 + 15 + 8 = 43 W
    power = compute_mechanical_power(torques, joint_vels)
    assert pytest.approx(power, rel=1e-5) == 43.0


def test_compute_mechanical_power_negative_work_clamped():
    """Verify negative mechanical work (braking) is clamped to zero (non-regen)."""
    torques = np.array([10.0, -10.0, 5.0])
    joint_vels = np.array([-2.0, 2.0, 3.0])
    # 10*(-2) = -20 -> clamped to 0
    # (-10)*2 = -20 -> clamped to 0
    # 5*3 = 15 -> 15
    power = compute_mechanical_power(torques, joint_vels)
    assert pytest.approx(power, rel=1e-5) == 15.0


def test_compute_cost_of_transport_known_values():
    """Verify CoT on known mass, power, and velocity."""
    mass = 12.0
    g = 9.81
    velocity = 0.5
    # Let CoT = 0.4 -> power = 0.4 * 12 * 9.81 * 0.5 = 23.544
    power = 23.544
    cot = compute_cost_of_transport(
        mean_mechanical_power=power,
        total_mass=mass,
        gravity=g,
        mean_forward_velocity=velocity,
    )
    assert pytest.approx(cot, rel=1e-4) == 0.40


def test_compute_cost_of_transport_zero_and_low_velocity():
    """Verify CoT returns NaN when robot is stationary or below minimum threshold."""
    mass = 12.0
    g = 9.81
    power = 10.0

    # Exactly zero velocity
    cot_zero = compute_cost_of_transport(
        mean_mechanical_power=power,
        total_mass=mass,
        gravity=g,
        mean_forward_velocity=0.0,
    )
    assert math.isnan(cot_zero), "CoT must be NaN for zero forward velocity"

    # Low creeping velocity below 0.05 m/s threshold
    cot_low = compute_cost_of_transport(
        mean_mechanical_power=power,
        total_mass=mass,
        gravity=g,
        mean_forward_velocity=0.03,
    )
    assert math.isnan(cot_low), "CoT must be NaN for velocity < threshold"


def test_compute_cost_of_transport_invalid_params():
    """Verify ValueError is raised on non-positive mass or gravity."""
    with pytest.raises(ValueError):
        compute_cost_of_transport(10.0, total_mass=-5.0, gravity=9.81, mean_forward_velocity=0.5)

    with pytest.raises(ValueError):
        compute_cost_of_transport(10.0, total_mass=12.0, gravity=0.0, mean_forward_velocity=0.5)


def test_evaluate_survival():
    """Verify survival condition: no torso contact AND sufficient distance."""
    # Robot fell (torso contact occurred)
    assert not evaluate_survival(has_torso_contact=True, distance_traveled=2.0)

    # Robot stood still or barely moved (< 0.5m)
    assert not evaluate_survival(has_torso_contact=False, distance_traveled=0.1)

    # Robot stayed upright and made meaningful forward progress
    assert evaluate_survival(has_torso_contact=False, distance_traveled=1.5)
