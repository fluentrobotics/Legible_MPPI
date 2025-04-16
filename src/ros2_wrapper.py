import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from tf2_wrapper import TF2Wrapper
import torch
from pytorch_mppi import MPPI
from config import *
from utils import dynamics, normalize_angle, save_data
import numpy as np
import math
import time
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.vectorized import contains
from sm_mppi import SMMPPIController




class MPPLocalPlannerMPPI(Node):
    def __init__(self):
        super().__init__('mpc_local_planner_mppi')

        # Initialize parameters

        self.tf2_wrapper = TF2Wrapper(self)
        self.rollouts = torch.zeros((7, NUM_SAMPLES, 2))
        self.costs = torch.zeros((7, NUM_SAMPLES, 2))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ROS2 setup
        self.cmd_vel_pub = self.create_publisher(TwistStamped, "/stretch/cmd_vel", 1)
        self.timer = self.create_timer(0.05, self.plan_and_publish)
        self.time_steps = []
        self.linear_velocities = []
        self.angular_velocities = []
        self.start_time = time.time()
        self.controller = SMMPPIController(STATIC_OBSTACLES, self.device)
        self.counter = 0

        self.current_state = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).to(self.device)  # [x, y, yaw]
        self.robot_velocity = torch.tensor([0.0, 0.0], dtype=torch.float32).to(self.device)
        self.previous_robot_state = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).to(self.device)

        self.agent_states = {i: torch.tensor([10.0, 10.0, 0.0], dtype=torch.float32).to(self.device) for i in range(ACTIVE_AGENTS)}
        self.agent_velocities = {i: torch.tensor([0.0, 0.0], dtype=torch.float32).to(self.device) for i in range(ACTIVE_AGENTS)}
        self.previous_agent_states = {i: torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).to(self.device) for i in range(ACTIVE_AGENTS)}

    def plan_and_publish(self):

        timer_callback_start_time = time.time()
        self.counter += 1
        T_map_baselink = self.tf2_wrapper.get_latest_pose("map", "base_link")
        if T_map_baselink is None:
            print("Can't find robot pose")
            return
            # Convert pose to MPPI state representation
        yaw = 2.0 * math.atan2(
            T_map_baselink.rotation.z, T_map_baselink.rotation.w
        )  # NOTE: assuming roll and pitch are negligible
        self.current_state = torch.tensor(
                [
                    T_map_baselink.translation.x,
                    T_map_baselink.translation.y,
                    yaw,
                ],
                dtype=torch.float32,
            ).to(self.device)
        
        if self.counter == 3:
            self.previous_robot_state = self.current_state
            
        if torch.any(self.previous_robot_state):
            self.robot_velocity = (self.current_state[:2] - self.previous_robot_state[:2])/HZ
            self.previous_robot_state = self.current_state
        for i in range(ACTIVE_AGENTS):
            tt = time.time()
            T_map_agent = self.tf2_wrapper.get_latest_pose("map", HUMAN_FRAME + "_" + str(i+1))
            #print("tt ", time.time() - tt)
            if T_map_agent is None:
                print("Can't find human pose") 
                agent_state = torch.tensor([10.0, 10.0, 0.0], dtype=torch.float32).to(self.device)    
            else:
                #print("found human ", i)
                yaw = 2.0 * math.atan2(
                    T_map_agent.rotation.z, T_map_agent.rotation.w
                )  # NOTE: assuming roll and pitch are negligible
                agent_state = torch.tensor([T_map_agent.translation.x, T_map_agent.translation.y, yaw], dtype=torch.float32).to(self.device)
                agent_state[2] = torch.arctan2(torch.sin(agent_state[2]), torch.cos(agent_state[2]))
                self.agent_states[i] = agent_state
                # Calculate velocity if previous state exists
            if i in self.previous_agent_states:
                prev_x, prev_y, prev_yaw = self.previous_agent_states[i]
                velocity = torch.tensor([
                            (agent_state[0] - prev_x) / HZ,
                            (agent_state[1] - prev_y) / HZ
                        ], dtype=torch.float32).to(self.device)

                self.agent_velocities[i] = velocity
            self.previous_agent_states[i] = agent_state


        # Compute optimal control using MPPI
        action, self.costs, self.rollouts, termination = self.controller.compute_control(
            self.current_state, self.previous_robot_state, self.robot_velocity, self.agent_states, self.previous_agent_states, self.agent_velocities
        )

        if action is not None and not termination:
            twist_stamped = TwistStamped()
            twist_stamped.header.stamp = self.get_clock().now().to_msg()
            twist_stamped.twist.linear.x = min(action[0].item(), VMAX)
            twist_stamped.twist.angular.z = action[1].item()
            self.cmd_vel_pub.publish(twist_stamped)
            self.linear_velocities.append(action[0].item())
            self.angular_velocities.append(action[1].item())
            self.time_steps.append(time.time() - self.start_time)
        elif termination:
            print("Reached the goal!!!!!")
            self.controller.move_to_next_goal()
            # self.save_plot()
        else:
            self.get_logger().warn("Failed to compute optimal controls")
        timer_callback_end_time = time.time()
        # print(timer_callback_end_time - timer_callback_start_time)


        
def main(args=None):
    rclpy.init(args=args)
    node = MPPLocalPlannerMPPI()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
