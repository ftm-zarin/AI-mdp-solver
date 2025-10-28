"""
solver.py

Contains the MDPSolver class, which implements the Value Iteration
and Policy Iteration algorithms.

The solver is initialized with an environment and parameters (gamma, epsilon)
and provides methods to solve for the optimal utilities and policy.
"""

import logging
import random
import copy

logger = logging.getLogger(__name__)

class MDPSolver:
    """
    Solves a given MDP environment using dynamic programming.
    
    Attributes:
        env: An object with the MDP interface (get_states, get_actions, etc.)
        gamma (float): The discount factor.
        epsilon (float): The convergence threshold.
        utilities (dict): A dictionary mapping {state: utility_value}.
    """
    def __init__(self, environment, gamma, epsilon):
        self.env = environment
        self.gamma = gamma
        self.epsilon = epsilon
        # Initialize utilities to 0 for all states
        self.utilities = {s: 0.0 for s in self.env.get_states()}
        logger.debug("MDPSolver initialized.")

    def solve_value_iteration(self):
        """
        Performs Value Iteration to find the optimal utilities.
        
        Algorithm:
        1. Start with U_0(s) = 0 for all s.
        2. Repeat for i = 0, 1, ...:
           U_{i+1}(s) = R(s) + gamma * max_a [ sum_{s'} T(s,a,s') * U_i(s') ]
        3. Stop when max |U_{i+1}(s) - U_i(s)| < epsilon.
        
        Returns:
            (dict, dict): A tuple of (final_utilities, optimal_policy)
        """
        iteration = 0
        while True:
            iteration += 1
            delta = 0.0
            # Create a new utility dict for the i+1 iteration
            new_utilities = copy.deepcopy(self.utilities)
            
            for state in self.env.get_states():
                # Terminal states have their reward as their utility
                if state in self.env.terminal_states:
                    new_utilities[state] = self.env.get_reward(state)
                    continue
                
                # Calculate the Bellman update: R(s) + gamma * max_a(Q(s,a))
                q_values = self._get_q_values(state, self.utilities)
                
                # max_a(Q(s,a)) is 0 if no actions are possible
                max_q = max(q_values.values()) if q_values else 0.0
                
                new_utilities[state] = self.env.get_reward(state) + (self.gamma * max_q)
                
                # Track the maximum change in utility for this iteration
                delta = max(delta, abs(new_utilities[state] - self.utilities[state]))

            # Update utilities for the next iteration
            self.utilities = new_utilities
            
            logger.debug(f"Value Iteration {iteration}: max delta = {delta:.6f}")
            
            # Check for convergence
            if delta < self.epsilon:
                logger.info(f"Value Iteration converged in {iteration} iterations.")
                break
                
        # Once utilities have converged, extract the optimal policy
        policy = self._extract_policy(self.utilities)
        return self.utilities, policy

    def solve_policy_iteration(self):
        """
        Performs Policy Iteration to find the optimal policy.
        
        Algorithm:
        1. Start with a random policy pi_0.
        2. Repeat:
           a) Policy Evaluation: Calculate U_i = U^{pi_i}
           b) Policy Improvement: pi_{i+1} = argmax_a [ sum_{s'} T(s,a,s') * U_i(s') ]
        3. Stop when pi_{i+1} == pi_i (policy is stable).
        
        Returns:
            (dict, dict): A tuple of (final_utilities, optimal_policy)
        """
        # 1. Start with a random policy
        policy = {}
        for state in self.env.get_states():
            if state in self.env.terminal_states:
                policy[state] = None
            else:
                actions = self.env.get_actions(state)
                policy[state] = random.choice(actions) if actions else None
        
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"Policy Iteration {iteration}: Starting...")
            
            # 2a. Policy Evaluation: Calculate utilities for the current policy
            logger.debug("Policy Evaluation step...")
            self.utilities = self._policy_evaluation(policy)
            
            # 2b. Policy Improvement: Find a new, better policy
            logger.debug("Policy Improvement step...")
            is_stable = True
            new_policy = {}
            
            for state in self.env.get_states():
                if state in self.env.terminal_states:
                    new_policy[state] = None
                    continue
                    
                # Find the action that maximizes expected utility (using current U)
                q_values = self._get_q_values(state, self.utilities)
                
                if not q_values:
                    new_policy[state] = None
                    continue

                best_action = max(q_values, key=q_values.get)
                new_policy[state] = best_action
                
                # Check if the policy has changed for this state
                if best_action != policy.get(state):
                    is_stable = False
            
            # Update the policy
            policy = new_policy
            
            # 3. Stop when policy is stable
            if is_stable:
                logger.info(f"Policy Iteration converged in {iteration} iterations.")
                break
                
        return self.utilities, policy

    def _policy_evaluation(self, policy):
        """
        Helper for Policy Iteration.
        Calculates the utility U(s) for a *fixed* policy.
        
        Solves the linear equations:
        U_i(s) = R(s) + gamma * sum_{s'} T(s, pi(s), s') * U_i(s')
        
        This is done iteratively, similar to Value Iteration.
        """
        eval_utilities = {s: 0.0 for s in self.env.get_states()} # Start from 0
        
        while True:
            delta = 0.0
            new_eval_utilities = copy.deepcopy(eval_utilities)
            
            for state in self.env.get_states():
                # Terminal states have their reward as their utility
                if state in self.env.terminal_states:
                    new_eval_utilities[state] = self.env.get_reward(state)
                    continue

                # Get the *only* action allowed by the policy
                action = policy.get(state)
                if action is None:
                    # State has no actions (e.g., non-terminal, but trapped)
                    new_eval_utilities[state] = self.env.get_reward(state)
                    continue
                
                # Calculate expected utility using ONLY the policy's action
                expected_utility = 0.0
                for prob, next_state in self.env.get_transitions(state, action):
                    expected_utility += prob * eval_utilities[next_state]
                
                # Bellman update for a fixed policy
                new_eval_utilities[state] = (
                    self.env.get_reward(state) + (self.gamma * expected_utility)
                )
                
                delta = max(delta, abs(new_eval_utilities[state] - eval_utilities[state]))

            eval_utilities = new_eval_utilities
            
            # Check for convergence
            if delta < self.epsilon:
                break
                
        return eval_utilities

    def _get_q_values(self, state, utilities):
        """
        Calculates the Q-value (expected utility) for all possible
        actions 'a' from a given 'state', using the given utilities 'U'.
        
        Q(s, a) = sum_{s'} T(s, a, s') * U(s')
        
        Returns:
            dict: A dictionary mapping {action: q_value}
        """
        q_values = {}
        for action in self.env.get_actions(state):
            expected_utility = 0.0
            for prob, next_state in self.env.get_transitions(state, action):
                expected_utility += prob * utilities[next_state]
            q_values[action] = expected_utility
        return q_values

    def _extract_policy(self, utilities):
        """
        Extracts the optimal policy from a converged utility function.
        
        pi*(s) = argmax_a [ sum_{s'} T(s, a, s') * U(s') ]
        
        Args:
            utilities (dict): The converged utility function {state: U(s)}.
            
        Returns:
            dict: The optimal policy {state: action}.
        """
        policy = {}
        for state in self.env.get_states():
            if state in self.env.terminal_states:
                policy[state] = None
                continue
            
            # Get Q-values for all actions from this state
            q_values = self._get_q_values(state, utilities)
            
            if not q_values:
                policy[state] = None # No actions possible
            else:
                # The best action is the one with the highest Q-value
                best_action = max(q_values, key=q_values.get)
                policy[state] = best_action
                
        return policy
