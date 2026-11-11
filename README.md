# Legible MPPI

Code accompanying [*Intent Representation and Human Distraction Modulate the Effects of Legible Robot Motion in Hallway Navigation*](https://arxiv.org/abs/2609.40158), by Pranav Goyal, Andrew Stratton, and Christoforos Mavrogiannis (2026).

The repository provides five controllers for social robot navigation using model predictive path integral control with [PyTorch-MPPI](https://github.com/UM-ARM-Lab/pytorch_mppi), alongside a configurable ROS 2 example.

| Algorithm | Controller |
| --- | --- |
| Social Momentum (SM) | [`SMMPPIController`](src/sm_mppi.py) |
| No legibility (NL) | [`NLMPPIController`](src/nl_mppi.py) |
| Goal-based legibility (GL) | [`GLMPPIController`](src/gl_mppi.py) |
| Passing-side legibility (PL) | [`PLMPPIController`](src/pl_mppi.py) |
| Dynamic passing-side legibility (DPL) | [`DPLMPPIController`](src/dpl_mppi.py) |

## Installation

From the repository root, using Python 3.10:

```bash
python3 -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## Usage

```python
import torch

from dpl_mppi import DPLMPPIController

controller = DPLMPPIController([], "cpu", goal=(0.0, -1.5))
action, rollouts, costs, reached = controller.compute_control(
    current_state=torch.tensor([0.0, 1.4, -torch.pi / 2]),
    robot_velocity=torch.tensor([0.0, -0.3]),
    agent_states={0: torch.tensor([0.2, 0.4, torch.pi / 2])},
    agent_velocities={0: torch.tensor([0.0, 0.3])},
)
```

States are `[x, y, yaw]`, world-frame velocities are `[vx, vy]`, and actions are `[linear_speed, yaw_rate]`, in meters, seconds, and radians. Use float32 tensors on the controller's device, at least one tracked pedestrian, and matching pedestrian IDs. Static obstacles are polygons in the same coordinate frame.

Stop when `reached` is true; use `set_goal((x, y))` to select the next destination. GL requires `hallway_goals=((x1, y1), (x2, y2))`, with `goal` set to one endpoint. PL requires `passing_side=-1` or `1`. Numerical settings are defined in [`MPPIConfig`](src/config.py).

## ROS 2

The example requires `rclpy`, `geometry_msgs`, and `tf2_ros`. Configure the hallway polygons in [`example_config.py`](src/example_config.py):

```bash
python3 src/ros2_wrapper.py --ros-args -p algorithm:=dpl
```

It uses `map` to `base_link` and `map` to `human_1` transforms and publishes `TwistStamped` to `/cmd_vel`. Configure frames, topic, and goal through the parameters in [`ros2_wrapper.py`](src/ros2_wrapper.py).

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{goyal2026rethinkinglegibility,
  title         = {Intent Representation and Human Distraction Modulate the Effects of Legible Robot Motion in Hallway Navigation},
  author        = {Pranav Goyal and Andrew Stratton and Christoforos Mavrogiannis},
  year          = {2026},
  eprint        = {2609.40158},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2609.40158}
}
```

## License

[BSD 3-Clause](LICENSE.md).
