"""
Procedural Terrain Generator and Heightfield Query Engine for Quadruped Locomotion.
Supports Flat, Incline Slope, Sinusoidal Rough, and Discrete Stair/Obstacle Terrains
with analytical surface height, contact normal, and friction evaluation.

Author: Youseef Abdel Tawwab
"""

import enum
from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


class TerrainType(enum.Enum):
    FLAT = "flat"
    SLOPE = "slope"
    ROUGH = "rough"
    STAIRS = "stairs"
    STEPPING_STONES = "stepping_stones"


@dataclass
class TerrainConfig:
    terrain_type: TerrainType = TerrainType.FLAT
    # Slope parameters
    pitch_angle: float = 0.0     # Inclination pitch in radians
    roll_angle: float = 0.0      # Inclination roll in radians
    # Rough terrain parameters (superposition of sinusoidal waves)
    roughness_amplitude: float = 0.035  # Max vertical variation (m)
    roughness_frequency: float = 1.5    # Spatial frequency (rad/m)
    num_octaves: int = 3                # Fractal octaves
    roughness_seed: int = 42
    # Stair parameters
    step_width: float = 0.35            # Tread depth (m)
    step_height: float = 0.04           # Riser height (m)
    # Friction parameters
    default_friction: float = 0.8
    friction_range: Tuple[float, float] = (0.4, 1.2)


class ProceduralTerrain:
    """
    Procedural terrain generator providing high-speed analytical surface queries
    for quadruped foot-ground contact simulation.
    """

    def __init__(self, config: Optional[TerrainConfig] = None):
        self.config = config if config is not None else TerrainConfig()
        self._rng = np.random.RandomState(self.config.roughness_seed)
        
        # Precompute random phase offsets for rough terrain octaves
        self._octave_phases = [
            (self._rng.uniform(0, 2 * np.pi), self._rng.uniform(0, 2 * np.pi))
            for _ in range(self.config.num_octaves)
        ]
        self._octave_weights = [0.5 ** i for i in range(self.config.num_octaves)]
        weight_sum = sum(self._octave_weights)
        self._octave_weights = [w / weight_sum for w in self._octave_weights]

        self.friction_coeff = self.config.default_friction

    def set_friction(self, friction: float) -> None:
        """Dynamically set ground friction coefficient for Sim-to-Real randomization."""
        self.friction_coeff = float(np.clip(friction, 0.05, 3.0))

    def randomize_friction(self) -> float:
        """Sample friction within configured domain randomization bounds."""
        low, high = self.config.friction_range
        self.friction_coeff = float(self._rng.uniform(low, high))
        return self.friction_coeff

    def get_height(self, x: float, y: float) -> float:
        """
        Analytical elevation z = f(x, y) at coordinates (x, y) in world frame.
        """
        t_type = self.config.terrain_type

        if t_type == TerrainType.FLAT:
            return 0.0

        elif t_type == TerrainType.SLOPE:
            # Linear incline plane: z = x * tan(pitch) + y * tan(roll)
            z = x * np.tan(self.config.pitch_angle) + y * np.tan(self.config.roll_angle)
            return float(z)

        elif t_type == TerrainType.ROUGH:
            # Multi-octave sinusoidal roughness
            z = 0.0
            for i, ((px, py), weight) in enumerate(zip(self._octave_phases, self._octave_weights)):
                freq = self.config.roughness_frequency * (2 ** i)
                z += weight * np.sin(freq * x + px) * np.cos(freq * y + py)
            return float(z * self.config.roughness_amplitude)

        elif t_type == TerrainType.STAIRS:
            # Ascending/descending stairs along positive x
            if x <= 0.0:
                return 0.0
            step_idx = np.floor(x / self.config.step_width)
            return float(step_idx * self.config.step_height)

        elif t_type == TerrainType.STEPPING_STONES:
            # Periodic stones separated by depressions
            grid_x = np.floor(x / 0.4)
            grid_y = np.floor(y / 0.4)
            rem_x = x - grid_x * 0.4
            rem_y = y - grid_y * 0.4
            if 0.05 <= rem_x <= 0.35 and 0.05 <= rem_y <= 0.35:
                # On stone
                return float(0.02 * np.sin(grid_x + grid_y))
            else:
                # Depression
                return -0.05

        return 0.0

    def get_normal(self, x: float, y: float, eps: float = 1e-4) -> np.ndarray:
        """
        Compute unit surface normal vector n = [-dz/dx, -dz/dy, 1] / ||...||
        using central numerical differentiation.
        """
        if self.config.terrain_type == TerrainType.FLAT:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)

        elif self.config.terrain_type == TerrainType.SLOPE:
            # Exact analytical normal for linear plane
            dz_dx = np.tan(self.config.pitch_angle)
            dz_dy = np.tan(self.config.roll_angle)
            n = np.array([-dz_dx, -dz_dy, 1.0], dtype=np.float64)
            return n / np.linalg.norm(n)

        # Central difference for general terrains
        dz_dx = (self.get_height(x + eps, y) - self.get_height(x - eps, y)) / (2.0 * eps)
        dz_dy = (self.get_height(x, y + eps) - self.get_height(x, y - eps)) / (2.0 * eps)

        n = np.array([-dz_dx, -dz_dy, 1.0], dtype=np.float64)
        norm = np.linalg.norm(n)
        return n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def check_foot_contact(
        self,
        foot_pos_world: np.ndarray,
        foot_radius: float = 0.022
    ) -> Tuple[bool, float, np.ndarray]:
        """
        Evaluate foot contact with terrain surface.
        
        Args:
            foot_pos_world: [x, y, z] center of spherical foot in world frame.
            foot_radius: radius of spherical foot (m).

        Returns:
            Tuple of:
            - in_contact (bool): True if lowest point of foot touches/penetrates ground.
            - penetration_depth (float): depth > 0 if penetrating ground, else clearance < 0.
            - normal (np.ndarray): unit ground normal at contact point.
        """
        x, y, z = foot_pos_world
        terrain_z = self.get_height(x, y)
        lowest_foot_z = z - foot_radius
        penetration = terrain_z - lowest_foot_z

        normal = self.get_normal(x, y)
        in_contact = penetration > 0.0

        return in_contact, float(penetration), normal
