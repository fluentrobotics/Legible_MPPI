from collections.abc import Sequence

import torch

from config import MPPIConfig
from legibility_mppi import LegibilityMPPIController


class GLMPPIController(LegibilityMPPIController):
    """Goal-based legibility over the two hallway endpoints."""

    def __init__(
        self,
        static_obs: Sequence[Sequence[tuple[float, float]]],
        device: torch.device | str,
        *,
        goal: Sequence[float],
        hallway_goals: Sequence[Sequence[float]],
        config: MPPIConfig | None = None,
    ) -> None:
        super().__init__(static_obs, device, goal=goal, config=config)
        self.candidate_goals = torch.tensor(
            hallway_goals, dtype=torch.float32, device=self.device
        )
        self.set_goal(goal)

    def set_goal(self, goal: Sequence[float]) -> None:
        super().set_goal(goal)
        self.other_goals = self.candidate_goals[
            torch.any(self.candidate_goals != self.goal, dim=-1)
        ]

    def terminal_cost(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        trajectories = state[0]
        return (
            10 * self.legibility_cost(trajectories)
            + 5 * self.collision_avoidance_cost(trajectories)
        )
