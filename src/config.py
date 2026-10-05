from dataclasses import dataclass


@dataclass(frozen=True)
class MPPIConfig:
    dt: float = 0.3
    horizon: int = 10
    num_samples: int = 250
    noise_variance: tuple[float, float] = (2.0, 2.0)
    control_min: tuple[float, float] = (0.0, -1.0)
    control_max: tuple[float, float] = (0.4, 1.0)
    temperature: float = 1.0
    goal_tolerance: float = 0.4
    interaction_distance: float = 5.0
    interaction_half_angle: float = 120.0
    obstacle_distance: float = 1.0
    history_length: int = 5
    goal_cost_scale: float = 5.6
