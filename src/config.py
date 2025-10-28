"""
config.py

Contains all configuration constants for the MDP Gridworld.
Separating constants here makes the environment code cleaner and
the parameters of the world easier to modify.
"""

import logging

# --- Grid Dimensions ---
GRID_HEIGHT = 3
GRID_WIDTH = 4

# --- Special States ---
# Tuples are (row, col)
WALL_STATES = {(1, 1)}
GOAL_STATE = (0, 3)
TRAP_STATE = (1, 3)
START_STATE = (2, 0) # Note: Not used by solver, but good for reference

# --- Rewards ---
REWARD_GOAL = 1.0
REWARD_TRAP = -1.0
REWARD_DEFAULT = -0.04  # The "living penalty" or cost of moving

# --- Actions and Transitions ---
# (dr, dc) for change in row, column
ACTIONS = {
    'North': (-1, 0),
    'South': (1, 0),
    'East':  (0, 1),
    'West':  (0, -1)
}

# Stochastic transition probabilities
PROB_INTEND = 0.8
PROB_SLIP_LEFT = 0.1  # 90 degrees left of intended action
PROB_SLIP_RIGHT = 0.1 # 90 degrees right of intended action

# --- Solver Parameters ---
DEFAULT_GAMMA = 0.99      # Discount factor
DEFAULT_EPSILON = 1e-4    # Convergence threshold for iteration algorithms

# --- Logging Configuration ---
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
DEFAULT_LOG_LEVEL = 'INFO'
