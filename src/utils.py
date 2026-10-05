import torch


def normalize_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def dynamics(state: torch.Tensor, action: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrate unicycle states (..., 3) with controls [speed, yaw rate]."""
    turn = action[..., 1] * dt
    radius = action[..., 0] / action[..., 1]
    local_x = torch.where(
        action[..., 1] == 0, action[..., 0] * dt, radius * torch.sin(turn)
    )
    local_y = torch.where(
        action[..., 1] == 0, 0.0, radius * (1.0 - torch.cos(turn))
    )
    heading = state[..., 2]
    return torch.stack(
        (
            state[..., 0] + local_x * torch.cos(heading) - local_y * torch.sin(heading),
            state[..., 1] + local_x * torch.sin(heading) + local_y * torch.cos(heading),
            normalize_angle(heading + turn),
        ),
        dim=-1,
    )
