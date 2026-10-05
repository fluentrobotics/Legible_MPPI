import torch
from torch.nn.functional import pad

from base_mppi import MPPIController
from config import MPPIConfig


class SMMPPIController(MPPIController):
    """Social Momentum MPPI."""

    default_config = MPPIConfig(control_min=(0.0, -1.5), control_max=(0.4, 1.5))

    def terminal_cost(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        trajectories = state[0]
        goal_cost = torch.linalg.vector_norm(
            trajectories[..., :2] - self.goal, dim=-1
        ).sum(dim=-1)
        social_cost = trajectories.new_zeros(trajectories.shape[0])
        for agent_id in self.interacting_agents:
            prediction = self.agent_predictions[agent_id]
            center = (trajectories[..., :2] + prediction) / 2
            robot_radius = pad(trajectories[..., :2] - center, (0, 1))
            agent_radius = pad(prediction - center, (0, 1))
            robot_velocity = pad(self.robot_velocity, (0, 1))
            agent_velocity = pad(self.agent_velocities[agent_id], (0, 1))
            momentum = (
                torch.cross(robot_radius, robot_velocity[None, None], dim=-1)
                + torch.cross(agent_radius, agent_velocity[None, None], dim=-1)
            )[..., 2]
            social_cost += torch.where(
                momentum[:, :-1] * momentum[:, 1:] > 0,
                -11 * momentum[:, :-1].abs(),
                10,
            ).sum(dim=-1)
        return (
            2 * goal_cost
            + social_cost
            + self.dynamic_obstacle_cost(trajectories)
            + 5 * self.collision_avoidance_cost(trajectories)
        )
