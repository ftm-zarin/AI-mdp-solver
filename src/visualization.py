"""
visualization.py

Contains helper functions for printing the state utilities and
the final policy in a human-readable grid format.
"""

def print_utilities_grid(utilities, gridworld, title="Utilities Grid"):
    """
    Prints a grid of utility values for each state.
    
    Args:
        utilities (dict): {state: utility_value}
        gridworld (Gridworld): The environment object, used for dimensions
                               and special states.
        title (str): The title to print above the grid.
    """
    print(title)
    print("-" * len(title))
    
    for r in range(gridworld.height):
        row_str = ""
        for c in range(gridworld.width):
            state = (r, c)
            if state in gridworld.walls:
                row_str += "  #WALL#  "
            elif state == gridworld.terminal_states:
                # Terminal states are in the utilities dict
                utility = utilities.get(state, 0.0)
                row_str += f" {utility: 8.2f} "
            elif state in utilities:
                utility = utilities[state]
                row_str += f" {utility: 8.2f} "
            else:
                row_str += "  ------  " # Should not happen
        print(row_str)
        print() # Extra space between rows

def print_policy_grid(policy, gridworld, title="Policy Grid"):
    """
    Prints a grid of actions (arrows) representing the policy.
    
    Args:
        policy (dict): {state: action}
        gridworld (Gridworld): The environment object.
        title (str): The title to print above the grid.
    """
    # Mapping from action names to arrow symbols
    action_symbols = {
        'North': '   ^    ',
        'South': '   v    ',
        'East':  '   >    ',
        'West':  '   <    ',
        None:    '   .    '  # For terminal states
    }
    
    print(title)
    print("-" * len(title))
    
    for r in range(gridworld.height):
        row_str = ""
        for c in range(gridworld.width):
            state = (r, c)
            if state in gridworld.walls:
                row_str += "  #WALL#  "
            elif state == gridworld.terminal_states:
                if state == gridworld.goal_state:
                     row_str += "  [GOAL]  "
                else:
                     row_str += "  [TRAP]  "
            elif state in policy:
                action = policy[state]
                symbol = action_symbols.get(action, '   ?    ')
                row_str += symbol
            else:
                row_str += "  ------  " # Should not happen
        print(row_str)
        print() # Extra space between rows
