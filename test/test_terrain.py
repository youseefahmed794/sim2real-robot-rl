"""
Unit tests for Procedural Terrain Generator and Heightfield Queries.

Author: Youseef Abdel Tawwab
"""

import numpy as np
import pytest
from sim2real_rl.envs.terrain import ProceduralTerrain, TerrainConfig, TerrainType


def test_flat_terrain():
    """Verify flat terrain returns zero elevation and vertical normal everywhere."""
    terrain = ProceduralTerrain(TerrainConfig(terrain_type=TerrainType.FLAT))

    for x in [-2.0, 0.0, 3.5]:
        for y in [-1.5, 0.0, 2.0]:
            h = terrain.get_height(x, y)
            assert h == 0.0, f"Expected 0.0 elevation, got {h}"

            n = terrain.get_normal(x, y)
            np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-6)


def test_slope_terrain():
    """Verify linear slope calculates elevation and normal consistent with incline angles."""
    pitch = np.radians(10.0)  # ~0.1745 rad
    roll = 0.0
    terrain = ProceduralTerrain(TerrainConfig(
        terrain_type=TerrainType.SLOPE,
        pitch_angle=pitch,
        roll_angle=roll
    ))

    # At x = 1.0, z = 1.0 * tan(pitch)
    expected_h = np.tan(pitch)
    h = terrain.get_height(1.0, 0.0)
    assert np.isclose(h, expected_h, atol=1e-5)

    # Unit normal vector verification
    n = terrain.get_normal(1.0, 0.0)
    assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-5)
    assert n[0] < 0  # Normal tilts back against positive slope


def test_rough_terrain():
    """Verify sinusoidal rough terrain produces bounded elevations and unit normals."""
    amp = 0.04
    terrain = ProceduralTerrain(TerrainConfig(
        terrain_type=TerrainType.ROUGH,
        roughness_amplitude=amp,
        roughness_frequency=2.0
    ))

    elevations = []
    for x in np.linspace(-3.0, 3.0, 15):
        for y in np.linspace(-3.0, 3.0, 15):
            h = terrain.get_height(x, y)
            elevations.append(h)
            assert abs(h) <= amp + 1e-4, f"Elevation {h} exceeded amplitude {amp}"

            n = terrain.get_normal(x, y)
            assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-4)

    # Verify terrain has non-zero spatial variation
    assert np.std(elevations) > 0.005


def test_stairs_terrain():
    """Verify stair terrain generates discrete rising treads."""
    step_w = 0.30
    step_h = 0.05
    terrain = ProceduralTerrain(TerrainConfig(
        terrain_type=TerrainType.STAIRS,
        step_width=step_w,
        step_height=step_h
    ))

    # Before stairs start (x <= 0)
    assert terrain.get_height(-0.5, 0.0) == 0.0

    # First step (0 < x < step_w)
    assert np.isclose(terrain.get_height(0.15, 0.0), 0.0, atol=1e-5)

    # Second step (step_w <= x < 2 * step_w)
    assert np.isclose(terrain.get_height(0.45, 0.0), step_h, atol=1e-5)


def test_foot_contact_detection():
    """Verify foot contact check and penetration calculation."""
    terrain = ProceduralTerrain(TerrainConfig(terrain_type=TerrainType.FLAT))
    foot_radius = 0.02

    # Foot in air: center at z = 0.10, lowest point at 0.08 > 0
    in_contact, depth, normal = terrain.check_foot_contact(
        np.array([0.0, 0.0, 0.10]), foot_radius=foot_radius
    )
    assert not in_contact
    assert depth < 0.0

    # Foot penetrating: center at z = 0.01, lowest point at -0.01, depth = +0.01
    in_contact, depth, normal = terrain.check_foot_contact(
        np.array([0.0, 0.0, 0.01]), foot_radius=foot_radius
    )
    assert in_contact
    assert np.isclose(depth, 0.01, atol=1e-5)
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-5)


def test_friction_randomization():
    """Verify ground friction setting and domain randomization."""
    terrain = ProceduralTerrain(TerrainConfig(friction_range=(0.3, 1.1)))
    assert terrain.friction_coeff == 0.8

    terrain.set_friction(0.55)
    assert terrain.friction_coeff == 0.55

    for _ in range(10):
        mu = terrain.randomize_friction()
        assert 0.3 <= mu <= 1.1
