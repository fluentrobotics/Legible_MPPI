import math
import torch
from pytorch_mppi import MPPI

import rclpy
import rclpy.time
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from tf2_wrapper import TF2Wrapper
from vis_utils import VisualizationUtils
from config import *
from geometry_msgs.msg import TwistStamped

import numpy as np

def normalize_angle(theta: torch.Tensor) -> torch.Tensor:
    """Normalize an angle to [-pi, pi]"""
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def dynamics(s: torch.Tensor, a: torch.Tensor, t=None) -> torch.Tensor:
    """
    Input:
    s: robot global state  (shape: BS x 3)
    a: robot action   (shape: BS x 2)


    Output:
    next robot global state after executing action (shape: BS x 3)
    """
    assert s.ndim == 2 and s.shape[-1] == 3
    assert a.ndim == 2 and a.shape[-1] == 2

    dt = DT

    s2_ego = torch.zeros_like(s)
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

    #print("S2 GLOBAL SHAPE: ", s2_global.shape)

    return s2_global

def save_data(counter, robot_position, human_positions, hst_predictions=None, hst_probabilities=None, cohan_predictions=None):
    pickle_file_path = PICKLE_DIR_PATH / "evaluation_data_multi.pkl"
    if hst_predictions is not None:
        hst_predictions = np.squeeze(hst_predictions, axis=0)

    data_to_save = {
        "cohan_predictions": cohan_predictions,
        "hst_predictions": hst_predictions,
        "hst_probabilities": hst_probabilities,
        "human_T": human_positions,
        "robot_T": robot_position
    }

    if self.counter == 0:
        write_mod = 'wb'
    else:
        write_mod = 'ab'

    with open(pickle_file_path.as_posix(), write_mod) as pickle_hd:
        pickle.dump(data_to_save, pickle_hd)
        print(f"Dump pickle at step {self.counter}")
    counter += 1
    return counter