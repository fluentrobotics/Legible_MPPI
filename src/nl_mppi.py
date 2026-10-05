import torch

from base_mppi import MPPIController


class NLMPPIController(MPPIController):
    """No-legibility MPPI with constant-velocity pedestrian predictions."""

    def terminal_cost(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        trajectories = state[0]
        goal_cost = torch.linalg.vector_norm(
            trajectories[..., :2] - self.goal, dim=-1
        ).sum(dim=-1)
        return (
            goal_cost
            + 3 * self.dynamic_obstacle_cost(trajectories)
            + 5 * self.collision_avoidance_cost(trajectories)
        )
