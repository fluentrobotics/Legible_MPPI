from collections.abc import Sequence
from typing import Literal

import torch

from config import MPPIConfig
from legibility_mppi import PassingSideMPPIController


class PLMPPIController(PassingSideMPPIController):
    """Passing-side legibility with an explicitly selected side."""

    def __init__(
        self,
        static_obs: Sequence[Sequence[tuple[float, float]]],
        device: torch.device | str,
        *,
        goal: Sequence[float],
        passing_side: Literal[-1, 1],
        goal_offset: tuple[float, float] = (0.15, 0.0),
        config: MPPIConfig | None = None,
    ) -> None:
        self.passing_side = passing_side
        super().__init__(
            static_obs, device, goal=goal, goal_offset=goal_offset, config=config
        )

    def set_goal(self, goal: Sequence[float]) -> None:
        super().set_goal(goal)
        index = 0 if self.passing_side == -1 else 1
        self.goal = self.candidate_goals[index].clone()
        self.other_goals = self.candidate_goals[[1 - index, 2]]

    def terminal_cost(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        trajectories = state[0]
        goal_cost = (
            torch.linalg.vector_norm(trajectories[..., :2] - self.goal, dim=-1)
            / self.config.goal_cost_scale
        ).sum(dim=-1)
        navigation_cost = trajectories.new_zeros(trajectories.shape[0])
        for agent_id in self.agent_predictions:
            if agent_id in self.interacting_agents:
                navigation_cost += 10 * self.legibility_cost(trajectories)
            else:
                navigation_cost += 5 * goal_cost
        return (
            navigation_cost
            + self.dynamic_obstacle_cost(trajectories)
            + 5 * self.collision_avoidance_cost(trajectories)
        )
