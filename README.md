# Legible_MPPI
MPPI Implementation for various legible algorithms for social robot navigation.

The implmentation is based on the [Pytorch-MPPI](https://github.com/UM-ARM-Lab/pytorch_mppi/tree/master) package. The algorithims included are:

- [Social Momentum](https://github.com/fluentrobotics/Legible_MPPI/blob/main/src/sm_mppi.py)
-  ~~DS Legibility~~
-  ~~DS Legibility (passing side goal)~~
-  ~~DS Legibility (passing side dynamic goals)~~
-  ~~Vanilla MPPI + Constant Velocity predictions~~

# Installation
```shell
pip install pytorch-mppi
```

An example on how to use the MPPIController for an algorithim with ROS2 can be found in [ros2_wrapper.py](https://github.com/fluentrobotics/Legible_MPPI/blob/main/src/ros2_wrapper.py). Note that this example is based on a Stretch RE2 robot with the human poses obtained from motion capture and published on the ROS2 tf.
