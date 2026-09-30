# Legible MPPI

Code accompanying *Rethinking Legibility in Social Robot Hallway Navigation: Impact of Intent Representation and Human Distraction* by Pranav Goyal, Andrew Stratton, and Christoforos Mavrogiannis. The paper studies how robot intent representation and pedestrian attention affect coordination in head-on hallway encounters. The controllers use model predictive path integral (MPPI) control built on [PyTorch-MPPI](https://github.com/UM-ARM-Lab/pytorch_mppi).

**Work in progress:** This repository currently includes the [Social Momentum controller](src/sm_mppi.py) and a [ROS 2 example](src/ros2_wrapper.py). The goal-based legibility (GL), passing-side legibility (PL), dynamic passing-side legibility (DPL), and no-legibility (NL) controllers from the paper will be added.

## Setup

Use a ROS 2 Python environment with message and TF2 packages, then install the Python dependencies:

```bash
pip install torch pytorch-mppi numpy shapely
```

The example expects `map` to `base_link` and `map` to `human_1` TF transforms and publishes `TwistStamped` commands to `/stretch/cmd_vel`. Set the hallway geometry and goals in [`src/config.py`](src/config.py) for your environment, then run:

```bash
python3 src/ros2_wrapper.py
```

## License

[BSD 3-Clause](LICENSE.md).
