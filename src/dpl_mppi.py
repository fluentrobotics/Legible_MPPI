import torch

from legibility_mppi import PassingSideMPPIController


class DPLMPPIController(PassingSideMPPIController):
    """Dynamic passing-side legibility using predicted pedestrian motion."""

    def terminal_cost(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        trajectories = state[0]
        goal_cost = (
            torch.linalg.vector_norm(trajectories[..., :2] - self.nominal_goal, dim=-1)
            / self.config.goal_cost_scale
        ).sum(dim=-1)
        navigation_cost = trajectories.new_zeros(trajectories.shape[0])
        for agent_id, prediction in self.agent_predictions.items():
            if agent_id in self.interacting_agents:
                distances = torch.linalg.vector_norm(
                    prediction[:, None, :] - self.candidate_goals[None, :2], dim=-1
                ).sum(dim=0)
                index = int(torch.argmin(torch.exp(-distances)))
                self.goal = self.candidate_goals[index].clone()
                self.other_goals = self.candidate_goals[[1 - index, 2]]
                navigation_cost += 10 * self.legibility_cost(trajectories)
            else:
                navigation_cost += 5 * goal_cost
        return (
            navigation_cost
            + self.dynamic_obstacle_cost(trajectories)
            + 5 * self.collision_avoidance_cost(trajectories)
        )
