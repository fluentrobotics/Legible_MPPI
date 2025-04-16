import numpy as np
import torch
from datetime import datetime

NOW = datetime.now()

VMAX = 0.33
DT = 0.3
HORIZON_LENGTH = 10
MAX_AGENT_NUM = 14
ACTIVE_AGENTS = 1
AGENT_GOALS = np.array([[0.0,-2.35], [0.0, 2.75]]) #goal of the agents, only required for evaluation metrics
HZ = 0.05 #frequency at which the controller should run
ANCA_CONTROL_POINTS = 5
USE_TERMINAL_COST = True
ANGULAR_THRESHOLD = 0.3
STATIC_OBSTACLES = [
            [(-0.7, -2.0), (-2.0, -2.0), (-2.0, 2.0), (-0.7, 2.0)],  
            [(0.7, -2.0), (2.0, -2.0), (2.0, 2.0), (0.7, 2.0)]  
            ]  # polgon shape of the static obstacles

NUM_SAMPLES = 250
NEED_ODOM = False
NEED_LASER = False
HUMAN_FRAME = "human" # for the TF transform from the motion capture
RADIUS = 0.2
NUM_CYCLES = 1 # Number of time to cycle between the goals
GOALS = np.array([[0.0, -1.5],[0.0, 1.5]])
REPEAT_GOALS = True
TERMINATION_TOLERANCE = 0.5