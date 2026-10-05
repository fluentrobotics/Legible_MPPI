import math
from dataclasses import replace

import rclpy
import torch
from geometry_msgs.msg import Transform, TwistStamped
from rclpy.node import Node

from dpl_mppi import DPLMPPIController
from example_config import STATIC_OBSTACLES
from gl_mppi import GLMPPIController
from nl_mppi import NLMPPIController
from pl_mppi import PLMPPIController
from sm_mppi import SMMPPIController
from tf2_wrapper import TF2Wrapper


class MPPLocalPlannerMPPI(Node):
    def __init__(self) -> None:
        super().__init__("mppi_local_planner")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("algorithm", "sm"),
                ("world_frame", "map"),
                ("robot_frame", "base_link"),
                ("agent_frames", ["human_1"]),
                ("cmd_vel_topic", "/cmd_vel"),
                ("control_period", 0.05),
                ("max_linear_speed", 0.33),
                ("goal", [0.0, -1.5]),
                ("passing_side", 0),
                ("device", "cpu"),
                ("dt", 0.3),
                ("horizon", 10),
                ("num_samples", 250),
                ("goal_tolerance", 0.4),
            ],
        )
        self.world_frame = self.get_parameter("world_frame").value
        self.robot_frame = self.get_parameter("robot_frame").value
        self.agent_frames = self.get_parameter("agent_frames").value
        self.max_linear_speed = self.get_parameter("max_linear_speed").value
        self.control_period = self.get_parameter("control_period").value
        self.device = torch.device(self.get_parameter("device").value)
        algorithm = self.get_parameter("algorithm").value
        controller_class = {
            "sm": SMMPPIController,
            "nl": NLMPPIController,
            "gl": GLMPPIController,
            "pl": PLMPPIController,
            "dpl": DPLMPPIController,
        }[algorithm]
        config = replace(
            controller_class.default_config,
            dt=self.get_parameter("dt").value,
            horizon=self.get_parameter("horizon").value,
            num_samples=self.get_parameter("num_samples").value,
            goal_tolerance=self.get_parameter("goal_tolerance").value,
        )
        goal = self.get_parameter("goal").value
        if algorithm == "gl":
            self.declare_parameter("hallway_goals", [0.0, -1.5, 0.0, 1.5])
            hallway_goals = self.get_parameter("hallway_goals").value
            self.controller = GLMPPIController(
                STATIC_OBSTACLES,
                self.device,
                goal=goal,
                hallway_goals=(hallway_goals[:2], hallway_goals[2:]),
                config=config,
            )
        elif algorithm in ("pl", "dpl"):
            self.declare_parameter(
                "goal_offset", [0.15, 0.0] if algorithm == "pl" else [0.2, 0.0]
            )
            goal_offset = tuple(self.get_parameter("goal_offset").value)
            if algorithm == "pl":
                passing_side = self.get_parameter("passing_side").value
                if passing_side == 0:
                    passing_side = 2 * int(torch.randint(0, 2, ())) - 1
                self.controller = PLMPPIController(
                    STATIC_OBSTACLES,
                    self.device,
                    goal=goal,
                    goal_offset=goal_offset,
                    passing_side=passing_side,
                    config=config,
                )
            else:
                self.controller = controller_class(
                    STATIC_OBSTACLES,
                    self.device,
                    goal=goal,
                    goal_offset=goal_offset,
                    config=config,
                )
        else:
            self.controller = controller_class(
                STATIC_OBSTACLES, self.device, goal=goal, config=config
            )
        self.tf2_wrapper = TF2Wrapper(self)
        self.cmd_vel_pub = self.create_publisher(
            TwistStamped, self.get_parameter("cmd_vel_topic").value, 1
        )
        self.previous_robot_state: torch.Tensor | None = None
        self.previous_agent_states: dict[int, torch.Tensor] = {}
        self.timer = self.create_timer(self.control_period, self.plan_and_publish)

    def state_from_transform(self, transform: Transform) -> torch.Tensor:
        return torch.tensor(
            [
                transform.translation.x,
                transform.translation.y,
                2 * math.atan2(transform.rotation.z, transform.rotation.w),
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def plan_and_publish(self) -> None:
        transforms = [
            self.tf2_wrapper.get_latest_pose(self.world_frame, frame)
            for frame in [self.robot_frame, *self.agent_frames]
        ]
        if any(transform is None for transform in transforms):
            self.publish_command(0.0, 0.0)
            self.previous_robot_state = None
            self.previous_agent_states.clear()
            self.get_logger().warning("Waiting for robot and pedestrian transforms.")
            return

        current_state = self.state_from_transform(transforms[0])
        agent_states = {
            agent_id: self.state_from_transform(transform)
            for agent_id, transform in enumerate(transforms[1:])
        }
        robot_velocity = torch.zeros(2, device=self.device)
        agent_velocities = {
            agent_id: torch.zeros(2, device=self.device) for agent_id in agent_states
        }
        if self.previous_robot_state is not None:
            robot_velocity = (
                current_state[:2] - self.previous_robot_state[:2]
            ) / self.control_period
            agent_velocities = {
                agent_id: (state[:2] - self.previous_agent_states[agent_id][:2])
                / self.control_period
                for agent_id, state in agent_states.items()
            }
        self.previous_robot_state = current_state
        self.previous_agent_states = agent_states
        action, _, _, reached = self.controller.compute_control(
            current_state, robot_velocity, agent_states, agent_velocities
        )
        if reached:
            self.publish_command(0.0, 0.0)
            self.get_logger().info("Goal reached.")
            self.timer.cancel()
        else:
            self.publish_command(
                min(float(action[0]), self.max_linear_speed), float(action[1])
            )

    def publish_command(self, speed: float, yaw_rate: float) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.robot_frame
        message.twist.linear.x = speed
        message.twist.angular.z = yaw_rate
        self.cmd_vel_pub.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MPPLocalPlannerMPPI()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_command(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
