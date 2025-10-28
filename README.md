# AI-Powered MDP Solver

A Python-based solver for Markov Decision Processes (MDPs) using Value Iteration and Policy Iteration. This project implements the classic 4x3 "Russell-Norvig" stochastic gridworld from scratch. It was developed as a core project for an Artificial Intelligence course.

## Features

* **Modular & Object-Oriented:** Code is cleanly separated into an environment (`environment.py`), a solver class (`solver.py`), and configuration (`config.py`).
* **Two Core DP Algorithms:** Implements both **Value Iteration** and **Policy Iteration** from first principles.
* **Stochastic Transitions:** The agent's actions are non-deterministic (80% chance of intended move, 10% slip left, 10% slip right), as per the classic problem definition.
* **Configurable Environment:** All environment parameters (grid size, rewards, wall locations, transition probabilities) are externalized in `src/config.py` for easy modification.
* **CLI Control:** Uses `argparse` to select the algorithm, discount factor ($\gamma$), and convergence epsilon ($\epsilon$).
* **Robust Logging:** Implements Python's `logging` module to log detailed solver steps to both the console and a file (`mdp_solver.log`).

## Core Concepts & Techniques

* **Markov Decision Processes (MDPs):** The formal framework used to model the problem, defined by (States, Actions, Transitions, Rewards).
* **Dynamic Programming:** The foundational technique used by both algorithms to find optimal solutions by breaking the problem into subproblems.
* **Bellman Equations:** The project implements the core Bellman update for Value Iteration and the Bellman equations for a fixed policy in Policy Evaluation.
    * **Value Iteration Update:** $U_{i+1}(s) \leftarrow R(s) + \gamma \max_{a} \sum_{s'} T(s, a, s') U_i(s')$
    * **Policy Evaluation Update:** $U_i(s) \leftarrow R(s) + \gamma \sum_{s'} T(s, \pi(s), s') U_i(s')$
* **Utility Functions:** Calculates the long-term expected utility $U(s)$ for every state.
* **Optimal Policies:** Extracts the optimal policy $\pi^*(s)$ that maximizes the expected utility from any given state.

---

## How It Works

This project is built around a clean, object-oriented structure that separates the problem's *definition* (the environment) from the *solution* (the solver).

### 1. Core Logic & Project Components

* **`src/config.py`:** This file acts as the "control panel" for the entire project. It defines all constants, such as grid dimensions, the location of walls and terminal states, all reward values, and the stochastic transition probabilities.
* **`src/environment.py`:** This file defines the `Gridworld` class. This class is the formal MDP implementation. It uses the constants from `config.py` to provide a clean interface to the solver, with methods like `get_states()`, `get_actions(state)`, `get_reward(state)`, and `get_transitions(state, action)`.
* **`src/solver.py`:** This is the heart of the project. The `MDPSolver` class is initialized with an `environment` object. It contains the core logic for both `solve_value_iteration()` and `solve_policy_iteration()`. This class is responsible for iterating until convergence and finding the final utility function.
* **`main.py`:** This is the main executable. It handles parsing command-line arguments (like which algorithm to run), sets up the file/console logging, creates the `Gridworld` and `MDPSolver` instances, and calls the appropriate solver method.
* **`src/visualization.py`:** This helper module contains functions (`print_utilities_grid`, `print_policy_grid`) to display the final results in a clean, human-readable format in the console.

### 2. Algorithms Implemented

#### Value Iteration
The `solve_value_iteration()` method finds the optimal utility function $U(s)$ by iteratively applying the Bellman update.

1.  It initializes $U_0(s) = 0$ for all states.
2.  It then enters a loop, calculating the next utility function $U_{i+1}$ from the previous $U_i$.
3.  For each state $s$, it calculates: $U_{i+1}(s) \leftarrow R(s) + \gamma \max_{a} \sum_{s'} T(s, a, s') U_i(s')$.
    * The $\max_{a}$ term is found by first calculating the Q-value $Q(s, a) = \sum_{s'} T(s, a, s') U_i(s')$ for all possible actions $a$ from state $s$.
4.  The algorithm tracks the maximum change in utility ($\delta$) during an iteration.
5.  The loop terminates when $\delta < \epsilon$, meaning the utilities have stabilized.
6.  Finally, it calls `_extract_policy()` to find the optimal policy $\pi^*(s)$ based on the converged utilities.

#### Policy Iteration
The `solve_policy_iteration()` method finds the optimal policy $\pi(s)$ by alternating between two steps.

1.  It starts with a random policy $\pi_0$.
2.  **Policy Evaluation:** The method `_policy_evaluation()` is called to calculate the utility function $U^{\pi_i}$ for the *current* policy $\pi_i$. This is itself an iterative process (similar to Value Iteration) that solves the simplified Bellman equation $U_i(s) = R(s) + \gamma \sum_{s'} T(s, \pi(s), s') U_i(s')$ until $U_i$ converges.
3.  **Policy Improvement:** The solver then uses the new utilities $U^{\pi_i}$ to find a (potentially) better policy $\pi_{i+1}$. For each state $s$, it finds the action $a$ that maximizes the expected utility: $\pi_{i+1}(s) = \arg\max_{a} \sum_{s'} T(s, a, s') U^{\pi_i}(s')$.
4.  If $\pi_{i+1} = \pi_i$, the policy is stable and the algorithm terminates. Otherwise, it sets $\pi_i \leftarrow \pi_{i+1}$ and loops back to step 2.

---

## Project Structure
 ```bash
ai-mdp-solver/
├── .gitignore         # Ignores Python cache and log files
├── LICENSE            # MIT License file
├── README.md          # This documentation file
├── main.py            # Main executable script with argument parsing
└── src/
    ├── __init__.py    # Makes 'src' a Python package
    ├── config.py      # Contains all environment constants (rewards, grid size, etc.)
    ├── environment.py # Defines the Gridworld class (the MDP itself)
    ├── solver.py      # Defines the MDPSolver class (Value/Policy Iteration)
    └── visualization.py # Helper functions to print the final grids
   ```

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/ftm-zarin/ai-mdp-solver.git](https://github.com/ftm-zarin/ai-mdp-solver.git)
    cd ai-mdp-solver
    ```

2.  **Run the Script:**
    You can run the solver using `main.py`. Use the `--algorithm` flag to choose your method.

3.  **Example Usage:**

    **Run Value Iteration (default):**
    ```bash
    python main.py -a value_iteration
    ```

    **Run Policy Iteration with a different gamma and verbose logging:**
    ```bash
    python main.py --algorithm policy_iteration --gamma 0.9 --log_level DEBUG
    ```

    **Expected Output (for Value Iteration):**
    ```
    ========================================
    Final Utilities
    ---------------

     0.8744    0.9177    0.9839    1.0000

     0.8066    #WALL#    0.6908   -1.0000

     0.7490    0.6970    0.6508    0.3952


    ----------------------------------------

    Policy Grid
    -----------

       >         >         >       [GOAL]

       ^       #WALL#      ^       [TRAP]

       ^         <         <         <

    ========================================
    ```

---

## Author

Feel free to connect or reach out if you have any questions!

* **Fatemeh Zarinjouei**
* **GitHub:** [@ftm-zarin](https://github.com/ftm-zarin)
* **Email:** [ftm.zariin@gmail.com](mailto:ftm.zariin@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
