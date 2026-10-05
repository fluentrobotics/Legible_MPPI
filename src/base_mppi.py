from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import partial

import torch
from pytorch_mppi import MPPI
from shapely import contains_xy
from shapely.geometry import MultiPolygon, Polygon

from config import MPPIConfig
from utils import dynamics


class MPPIController(ABC):
    default_config = MPPIConfig()

    def __init__(
        self,
        static_obs: Sequence[Sequence[tuple[float, float]]],
        device: torch.device | str,
        *,
        goal: Sequence[float],
        config: MPPIConfig | None = None,
    ) -> None:
        self.config = config if config is not None else self.default_config
        config = self.config
        self.device = torch.device(device)
        self.goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
        self.obstacles = MultiPolygon([Polygon(vertices) for vertices in static_obs])
        self.robot_velocity = torch.zeros(2, device=self.device)
        self.agent_velocities: Mapping[int, torch.Tensor] = {}
        self.agent_predictions: dict[int, torch.Tensor] = {}
        self.interacting_agents: set[int] = set()
        self.prediction_times = torch.linspace(
            config.dt, config.horizon * config.dt, config.horizon, device=self.device
        )
        self.mppi = MPPI(
            partial(dynamics, dt=config.dt),
            self.cost,
            3,
            torch.diag(torch.tensor(config.noise_variance, device=self.device)),
            num_samples=config.num_samples,
            horizon=config.horizon,
            device=self.device,
            lambda_=config.temperature,
            terminal_state_cost=self.terminal_cost,
            u_min=torch.tensor(config.control_min, device=self.device),
            u_max=torch.tensor(config.control_max, device=self.device),
        )

    def set_goal(self, goal: Sequence[float]) -> None:
        self.goal = torch.tensor(goal, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def compute_control(
        self,
        current_state: torch.Tensor,
        robot_velocity: torch.Tensor,
        agent_states: Mapping[int, torch.Tensor],
        agent_velocities: Mapping[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Return action (2,), rollouts (K, T, 3), costs (K,), and goal reached."""
        self.robot_velocity = robot_velocity
        self.agent_velocities = agent_velocities
        self.agent_predictions = {
            agent_id: state[:2]
            + self.prediction_times[:, None] * agent_velocities[agent_id]
            for agent_id, state in agent_states.items()
        }
        self.interacting_agents = set()
        for agent_id, state in agent_states.items():
            displacement = state[:2] - current_state[:2]
            bearing = torch.rad2deg(
                torch.atan2(displacement[1], displacement[0]) - current_state[2]
            )
            bearing = (bearing + 180) % 360 - 180
            if (
                torch.linalg.vector_norm(displacement)
                < self.config.interaction_distance
                and bearing.abs() <= self.config.interaction_half_angle
            ):
                self.interacting_agents.add(agent_id)

        action = self.mppi.command(current_state)
        self.mppi.u_init = action.clone()
        reached = bool(
            torch.linalg.vector_norm(current_state[:2] - self.goal)
            < self.config.goal_tolerance
        )
        return (
            action,
            self.mppi.states[0],
            self.mppi.cost_total,
            reached,
        )

    def cost(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return state.new_zeros(state.shape[:-1])

    @abstractmethod
    def terminal_cost(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Evaluate MPPI trajectories with shape (1, K, T, 3)."""

    def collision_avoidance_cost(self, state: torch.Tensor) -> torch.Tensor:
        positions = state[..., :2].detach().cpu().numpy()
        collisions = contains_xy(self.obstacles, positions[..., 0], positions[..., 1])
        return 10 * torch.as_tensor(
            collisions, dtype=state.dtype, device=state.device
        ).sum(dim=-1)

    def dynamic_obstacle_cost(self, state: torch.Tensor) -> torch.Tensor:
        cost = state.new_zeros(state.shape[:-2])
        for prediction in self.agent_predictions.values():
            distance = torch.linalg.vector_norm(state[..., :2] - prediction, dim=-1)
            cost += torch.where(
                distance < self.config.obstacle_distance, 1 / (1 + distance.square()), 0
            ).sum(dim=-1)
        return cost
