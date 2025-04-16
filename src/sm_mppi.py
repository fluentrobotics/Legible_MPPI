import torch
from pytorch_mppi import MPPI
from config import *
from utils import dynamics, normalize_angle, save_data
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.vectorized import contains

class SMMPPIController:
    def __init__(self,static_obs, device):
        # Initialize parameters from config
        self.horizon = HORIZON_LENGTH 
        self.dt = DT
        self.device = device
        self.angular_alignment_threshold = ANGULAR_THRESHOLD   # Angular error threshold in radians
        self.goals = torch.tensor(GOALS, dtype=torch.float32).to(self.device)
        self.rollouts = torch.zeros((7, NUM_SAMPLES, 2))
        self.costs = torch.zeros((7, NUM_SAMPLES, 2))
        self.max_cycles = NUM_CYCLES
        self.num_samples = NUM_SAMPLES
        self.polygons = [Polygon(obs) for obs in static_obs]
        self.multi_polygon = MultiPolygon(self.polygons)
        self.bounds = self.multi_polygon.bounds
        self.s2_ego = torch.zeros((self.num_samples, 3)).to(self.device)


        self.current_goal_index = 0
        self.cycle_count = 0
        self.counter = 0
        self.goal = self.goals[self.current_goal_index]
        self.agent_weights = {i: torch.tensor([0.1, 0.1, 0.0], dtype=torch.float32).to(self.device) for i in range(ACTIVE_AGENTS)}

        self.interacting_agents = []

        # Initialize MPPI with the dynamics and cost functions
        cov = torch.eye(2, dtype=torch.float32).to(self.device)
        cov[0, 0] = 2
        cov[1, 1] = 2
        
        # MPPI initialization
        cov = torch.eye(2, dtype=torch.float32).to(self.device)
        self.mppi = MPPI(
            self.dynamics,
            self.cost,
            3,  # State dimension
            cov,
            num_samples=self.num_samples,
            horizon=self.horizon,
            device=self.device,
            terminal_state_cost=self.terminal_cost,
            step_dependent_dynamics=True,
            u_min=torch.tensor([0.0, -1.5], dtype=torch.float32).to(self.device),
            u_max=torch.tensor([0.4, 1.5], dtype=torch.float32).to(self.device),
        )

    def compute_control(self, current_state, previous_robot_state, robot_velocity, agent_states, previous_agent_states,agent_velocities): 
        self.current_state = current_state
        self.previous_robot_state = previous_robot_state
        self.robot_velocity = robot_velocity
        
        self.agent_states = agent_states
        self.previous_agent_states = previous_agent_states
        self.agent_velocities = agent_velocities
        action = self.mppi.command(current_state)
        self.mppi.u_init = action
        rollouts = self.mppi.states.squeeze(0)
        costs = self.mppi.cost_total.squeeze(0)

        termination =  torch.linalg.norm(self.current_state[:2] - self.goal) < TERMINATION_TOLERANCE
        
        return action, rollouts, costs, termination

    def cost(self, state: torch.Tensor, action: torch.Tensor, t) -> torch.Tensor:
        """
        Cost function for MPPI optimization.
        Args:
            state: (num_samples, horizon, 3) - States over the horizon.
            action: (num_samples, horizon, 2) - Actions over the horizon.

        Returns:
            cost: (num_samples) - Total cost for each sample.
        """
        return 0

    def dynamics(self, s: torch.Tensor, a: torch.Tensor, t=None) -> torch.Tensor:
        """
        Input:
        s: robot global state  (shape: BS x 3)
        a: robot action   (shape: BS x 2)


        Output:
        next robot global state after executing action (shape: BS x 3)
        """
        assert s.ndim == 2 and s.shape[-1] == 3
        assert a.ndim == 2 and a.shape[-1] == 2

        dt = self.dt

        self.s2_ego.zero_()
        s2_ego = torch.zeros_like(s).to(self.device)
        s2_ego = self.s2_ego
        d_theta = a[:, 1] * dt
        turning_radius = a[:, 0] / a[:, 1]

        s2_ego[:, 0] = torch.where(
            a[:, 1] == 0, a[:, 0] * dt, turning_radius * torch.sin(d_theta)
        )
        s2_ego[:, 1] = torch.where(
            a[:, 1] == 0, 0.0, turning_radius * (1.0 - torch.cos(d_theta))
        )
        s2_ego[:, 2] = torch.where(a[:, 1] == 0, 0.0, d_theta)

        s2_global = torch.zeros_like(s)
        s2_global[:, 0] = (
            s[:, 0] + s2_ego[:, 0] * torch.cos(s[:, 2]) - s2_ego[:, 1] * torch.sin(s[:, 2])
        )
        s2_global[:, 1] = (
            s[:, 1] + s2_ego[:, 0] * torch.sin(s[:, 2]) + s2_ego[:, 1] * torch.cos(s[:, 2])
        )
        s2_global[:, 2] = normalize_angle(s[:, 2] + s2_ego[:, 2])

        return s2_global


    def terminal_cost(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        goal_expanded = self.goal[None, :]
        state_squeezed = state.squeeze()
        dist = torch.norm(goal_expanded-state_squeezed[:,:,:2], dim=2)# (N x T')
        goal_cost = torch.sum(dist, dim=1)
        dynamic_obstacle_costs = torch.zeros(self.num_samples).to(self.device)
        sm_costs = torch.zeros(self.num_samples).to(self.device)

        for i in range(ACTIVE_AGENTS):
            next_x_states = self.agent_states[i][0] + torch.linspace(self.dt,self.horizon*self.dt,self.horizon,device=self.device)* self.agent_velocities[i][0]
            next_y_states = self.agent_states[i][1] + torch.linspace(self.dt,self.horizon*self.dt,self.horizon,device=self.device)* self.agent_velocities[i][1]
            human_states  = torch.stack((next_x_states,next_y_states), dim=1)
            dist = torch.norm(state_squeezed[:,:,:2] -  human_states,dim=2)
            # social mometum cost
            sm_costs += self.SocialCost(state, i, human_states)
            #dynamic obstacle cost
            dynamic_obstacle_cost = torch.where(dist < 1.0, 1 / (1 + dist**2), torch.tensor(0.0, device=self.device))
            dynamic_obstacle_costs += torch.sum(dynamic_obstacle_cost,dim=1)
            #static obstacle cost
            static_costs = self.collision_avoidance_cost(state_squeezed)

        return 2*(goal_cost) + sm_costs + dynamic_obstacle_costs + 5*static_costs  #weights can be tuned for desired behaviour
        

    def get_interacting_agents(self):
        self.interacting_agents = []
        robot_state = self.current_state
        for idx, agent_state in self.agent_states.items():
            direction_to_agent = torch.arctan2(agent_state[1] - self.current_state[1], agent_state[0] - self.current_state[0])
            distance_to_agent = torch.norm(agent_state[:2] - self.current_state[:2])
            relative_angle = torch.rad2deg(direction_to_agent - robot_state[2])
            relative_angle = (relative_angle + 180) % 360 - 180  
            
            if -120 <= relative_angle <= 120 and distance_to_agent < 5.0:  
                self.interacting_agents.append(idx)
                self.agent_weights[idx] = (1/distance_to_agent)


    def SocialCost(self, state: torch.Tensor,i,human_states, **kwargs) -> torch.Tensor:
        sm_cost = 0.0
        self.get_interacting_agents()
        if i in self.interacting_agents:
            state_squeezed = state.squeeze()
            r_c = (state_squeezed[:,:,:2] + human_states) / 2
            r_ac = state_squeezed[:,:,:2] - r_c
            r_bc = human_states - r_c
            r_ac_3d = torch.nn.functional.pad(r_ac, (0, 1), "constant", 0)  # [N, T', 3]
            r_bc_3d = torch.nn.functional.pad(r_bc, (0, 1), "constant", 0)  # [N, T', 3]
            robot_velocity_3d = torch.nn.functional.pad(self.robot_velocity, (0, 1), "constant", 0)  # Shape: [3]
            agent_velocities_3d = torch.nn.functional.pad(self.agent_velocities[i], (0, 1), "constant", 0) 
            l_ab = torch.cross(r_ac_3d, robot_velocity_3d[None,None,:], dim=2) + torch.cross(r_bc_3d, agent_velocities_3d[None,None,:], dim=2)
            l_ab = l_ab[:, :, 2]
            l_ab_dot_product = l_ab[:, :-1] * l_ab[:, 1:]    # Determine if dot product is positive or not
            condition = l_ab_dot_product > 0  # Shape: [N, T'-1]
            l_ab_conditional = torch.where(condition, -11*torch.abs(l_ab[:, :-1]), torch.tensor(10.0, device=l_ab.device))  # Shape: [N, T'-1]
            sm_cost += torch.sum(l_ab_conditional, dim=1)
        return sm_cost
        
    def collision_avoidance_cost(self,state):
        xy_coords = state[..., :2].cpu().numpy()  # Shape: (N, T', 2)
        flattened_coords = xy_coords.reshape(-1, 2)
        x_min, y_min, x_max, y_max = self.bounds
        within_bounds = (
            (flattened_coords[:, 0] >= x_min) &
            (flattened_coords[:, 0] <= x_max) &
            (flattened_coords[:, 1] >= y_min) &
            (flattened_coords[:, 1] <= y_max)
        )

        # Use Shapely's vectorized `contains` for points within bounds
        collision_flags = contains(self.multi_polygon, flattened_coords[:, 0], flattened_coords[:, 1])
        collision_flags[~within_bounds] = False  # Points outside bounds are not collisions

        # Assign costs based on collision flags
        costs = torch.where(
            torch.tensor(collision_flags, dtype=torch.bool),
            torch.tensor(10.0),  
            torch.tensor(0.0) 
        )
        costs = costs.view(state.shape[0], state.shape[1]).sum(dim=1)

        return costs.to(state.device)

    def move_to_next_goal(self):
        if self.current_goal_index < len(self.goals) - 1:
            # Move to the next goal in the current cycle
            self.current_goal_index += 1
        else:
            # Finish the current cycle
            if self.cycle_count >= self.max_cycles:
                print(f"All {self.max_cycles} cycles completed! Stopping the robot.")
                self.timer.cancel() 
                rclpy.shutdown()
                return
            else:
                self.cycle_count += 1
                self.current_goal_index = 0

        # Update the current goal
        self.goal = self.goals[self.current_goal_index]
        print(f"Moving to goal {self.current_goal_index + 1}: {self.goal} (Cycle {self.cycle_count + 1}/{self.max_cycles})")



# def calculate_angle_to_goal(self, current_state, goal):
#     """
#     Calculate the angle difference required to face the goal.

#     Args:
#         current_state (torch.Tensor): Current robot state [x, y, yaw].
#         goal (torch.Tensor): Goal position [x, y].

#     Returns:
#         float: Angle difference in radians.
#     """
#     goal_vector = goal[:2] - current_state[:2] 
#     desired_yaw = torch.atan2(goal_vector[1], goal_vector[0]) - 0.3
#     angle_diff = desired_yaw - current_state[2]  
#     return torch.atan2(torch.sin(angle_diff),torch.cos(angle_diff))