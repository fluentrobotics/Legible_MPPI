from collections.abc import Sequence

import torch

from base_mppi import MPPIController
from config import MPPIConfig


class LegibilityMPPIController(MPPIController):
    def __init__(
        self,
        static_obs: Sequence[Sequence[tuple[float, float]]],
        device: torch.device | str,
        *,
        goal: Sequence[float],
        config: MPPIConfig | None = None,
    ) -> None:
        super().__init__(static_obs, device, goal=goal, config=config)
        self.history = torch.zeros(
            self.config.num_samples, self.config.history_length, 3, device=self.device
        )
        self.other_goals: torch.Tensor

    def legibility_cost(self, state: torch.Tensor) -> torch.Tensor:
        # Track per-sample history for the full-state path cost.
        self.history = torch.cat((self.history[:, 1:], state[:, :1]), dim=1)
        history_cost = torch.linalg.vector_norm(
            torch.diff(self.history, dim=1), dim=-1
        ).sum(dim=-1, keepdim=True)
        path_cost = torch.linalg.vector_norm(
            torch.diff(state, dim=1, prepend=state[:, :1]), dim=-1
        ).cumsum(dim=-1) + history_cost
        goal_distance = torch.linalg.vector_norm(self.goal - state[..., :2], dim=-1)
        total_cost = goal_distance.sum(dim=-1) + path_cost.sum(dim=-1)
        goal_probability = torch.exp(-goal_distance - path_cost)
        normalizer = goal_probability.clone()
        for other_goal in self.other_goals:
            distance = torch.linalg.vector_norm(other_goal - state[..., :2], dim=-1)
            normalizer += torch.exp(-distance - path_cost)
        probability = (1 / normalizer) * goal_probability
        time_steps = torch.arange(
            state.shape[1], device=state.device, dtype=state.dtype
        )
        weights = 1 - (time_steps + self.config.history_length) / (
            self.config.horizon + self.config.history_length
        )
        return -100 * (weights * probability).sum(dim=-1) + 0.1 * total_cost


class PassingSideMPPIController(LegibilityMPPIController):
    def __init__(
        self,
        static_obs: Sequence[Sequence[tuple[float, float]]],
        device: torch.device | str,
        *,
        goal: Sequence[float],
        goal_offset: tuple[float, float] = (0.2, 0.0),
        config: MPPIConfig | None = None,
    ) -> None:
        super().__init__(static_obs, device, goal=goal, config=config)
        self.goal_offset = torch.tensor(
            goal_offset, dtype=torch.float32, device=self.device
        )
        self.set_goal(goal)

    def set_goal(self, goal: Sequence[float]) -> None:
        super().set_goal(goal)
        self.nominal_goal = self.goal.clone()
        self.candidate_goals = torch.stack(
            (self.goal - self.goal_offset, self.goal + self.goal_offset, self.goal)
        )
        self.other_goals = self.candidate_goals[:2]
