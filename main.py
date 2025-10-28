"""
main.py

Main entry point for the AI MDP Solver.
This script parses command-line arguments, sets up logging, initializes the
Gridworld environment and the MDPSolver, and then runs the selected algorithm.
Finally, it prints the resulting utilities and policy to the console.

Usage:
    python main.py -a value_iteration --gamma 0.9 --epsilon 0.001
    python main.py --algorithm policy_iteration --log_level DEBUG
"""

import argparse
import logging
import sys
from src.environment import Gridworld
from src.solver import MDPSolver
from src.visualization import print_utilities_grid, print_policy_grid
from src.config import (
    DEFAULT_GAMMA, 
    DEFAULT_EPSILON, 
    LOG_LEVELS, 
    DEFAULT_LOG_LEVEL
)

def setup_logging(log_level_str):
    """Configures console and file logging."""
    log_level = LOG_LEVELS.get(log_level_str.upper(), logging.INFO)
    
    # Base configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - [%(levelname)s] - %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler('mdp_solver.log', mode='w'), # File logger
            logging.StreamHandler(sys.stdout)               # Console logger
        ]
    )
    # Suppress overly verbose libraries if any were added
    # logging.getLogger("some_library").setLevel(logging.WARNING)
    logging.info(f"Logging initialized at {log_level_str} level.")

def main():
    """
    Parses arguments, runs the selected MDP solving algorithm, and prints results.
    """
    parser = argparse.ArgumentParser(
        description="Solve a Gridworld MDP using Value or Policy Iteration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-a', '--algorithm',
        type=str,
        default='value_iteration',
        choices=['value_iteration', 'policy_iteration'],
        help="The solver algorithm to use."
    )
    parser.add_argument(
        '-g', '--gamma',
        type=float,
        default=DEFAULT_GAMMA,
        help="Discount factor (gamma) for future rewards."
    )
    parser.add_argument(
        '-e', '--epsilon',
        type=float,
        default=DEFAULT_EPSILON,
        help="Convergence threshold (epsilon) for utility updates."
    )
    parser.add_argument(
        '-l', '--log_level',
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=LOG_LEVELS.keys(),
        help="Set the logging output level."
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logging.info(f"Starting MDP Solver with algorithm: {args.algorithm}")
    
    try:
        # 1. Initialize the environment
        env = Gridworld()
        logging.debug("Gridworld environment initialized.")
        
        # 2. Initialize the solver
        solver = MDPSolver(
            environment=env, 
            gamma=args.gamma, 
            epsilon=args.epsilon
        )
        logging.debug(f"MDPSolver initialized with gamma={args.gamma}, epsilon={args.epsilon}")
        
        # 3. Run the selected algorithm
        if args.algorithm == 'value_iteration':
            logging.info("Running Value Iteration...")
            utilities, policy = solver.solve_value_iteration()
        else: # policy_iteration
            logging.info("Running Policy Iteration...")
            utilities, policy = solver.solve_policy_iteration()
            
        logging.info("Solver algorithm finished successfully.")
        
        # 4. Print the results
        print("\n" + "="*40)
        print_utilities_grid(utilities, env, title="Final Utilities")
        print("\n" + "-"*40 + "\n")
        print_policy_grid(policy, env, title="Optimal Policy")
        print("="*40)

    except Exception as e:
        logging.error(f"An unhandled exception occurred: {e}", exc_info=True)
        print(f"An error occurred. Check 'mdp_solver.log' for details.", file=sys.stderr)
        sys.exit(1) # Exit with a non-zero status code

    logging.info("Program terminated successfully.")

if __name__ == "__main__":
    main()
