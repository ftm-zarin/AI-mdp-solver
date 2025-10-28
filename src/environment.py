"""
environment.py

Defines the Gridworld class, which represents the MDP environment.
It provides methods to get states, actions, rewards, and transition
probabilities, abstracting the world's logic from the solver.
"""

import logging
from src.config import (
    GRID_HEIGHT, GRID_WIDTH, WALL_STATES, GOAL_STATE, TRAP_STATE,
    REWARD_GOAL, REWARD_TRAP, REWARD_DEFAULT,
    ACTIONS, PROB_INTEND, PROB_SLIP_LEFT, PROB_SLIP_RIGHT
)

logger = logging.getLogger(__name__)

class Gridworld:
    """
    Implements the 4x3 Russell-Norvig Gridworld MDP.
    
    Provides an interface for the solver with methods like:
    - get_states()
    - get_actions(state)
    - get_reward(state)
    - get_transitions(state, action)
    """
    def __init__(self):
        self.height = GRID_HEIGHT
        self.width = GRID_WIDTH
        self.walls = WALL_STATES
        self.terminal_states = {GOAL_STATE, TRAP_STATE}
        
        # Build the set of all valid states
        self.states = set()
        for r in range(self.height):
            for c in range(self.width):
                if (r, c) not in self.walls:
                    self.states.add((r, c))
        
        # Pre-calculate action "slips" (left/right of intended)
        self._action_slips = {
            'North': {'left': 'West', 'right': 'East'},
            'South': {'left': 'East', 'right': 'West'},
            'East':  {'left': 'North', 'right': 'South'},
            'West':  {'left': 'South', 'right': 'North'}
        }
        logger.debug("Gridworld initialized with %d states.", len(self.states))

    def get_states(self):
        """Returns the set of all valid states (r, c) tuples."""
        return self.states

    def get_actions(self, state):
        """
        Returns the list of possible actions from a given state.
        An empty list is returned if the state is terminal.
        """
        if state in self.terminal_states:
            return []  # No actions possible from a terminal state
        return list(ACTIONS.keys())

    def get_reward(self, state):
        """Returns the reward for being in a given state."""
        if state == GOAL_STATE:
            return REWARD_GOAL
        if state == TRAP_STATE:
            return REWARD_TRAP
        return REWARD_DEFAULT

    def get_transitions(self, state, action):
        """
        Calculates the transition probabilities from (state, action).
        
        Returns:
            A list of (probability, next_state) tuples.
        """
        if state in self.terminal_states:
            return [] # No transitions out of terminal states
        
        transitions = []
        
        # 1. Get intended action and the two "slip" actions
        action_intend = action
        action_left = self._action_slips[action]['left']
        action_right = self._action_slips[action]['right']
        
        # 2. Calculate next state for each possible outcome
        #    (prob, action_name)
        for prob, act in [(PROB_INTEND, action_intend),
                          (PROB_SLIP_LEFT, action_left),
                          (PROB_SLIP_RIGHT, action_right)]:
            
            # Get the resulting state from this action
            next_state = self._calculate_next_state(state, act)
            transitions.append((prob, next_state))
            
        # 3. Consolidate probabilities for the same next_state
        #    e.g., if slipping left and slipping right both hit a wall
        #    and result in staying in the same state.
        consolidated = {}
        for prob, s_prime in transitions:
            consolidated[s_prime] = consolidated.get(s_prime, 0.0) + prob
            
        return list(consolidated.items())

    def _calculate_next_state(self, state, action_name):
        """
        Helper function to get the deterministic next state
        given a state and a *single* action (no slips).
        Handles wall and boundary collisions.
        """
        if state in self.terminal_states:
            return state # Should not happen, but good check
            
        # Get (dr, dc) for the action
        dr, dc = ACTIONS[action_name]
        
        next_r, next_c = state[0] + dr, state[1] + dc
        
        next_state = (next_r, next_c)
        
        # Check for boundary collision or wall collision
        if (not (0 <= next_r < self.height and 0 <= next_c < self.width) or
                (next_state in self.walls)):
            return state  # Bumps into wall/boundary, stays in place
        
        return next_state
