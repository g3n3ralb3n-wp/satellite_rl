"""Gymnasium environment for satellite sensor tasking on a grid."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Optional, Any


class SatelliteSensorGridEnv(gym.Env):
    """
    Gymnasium environment for satellite sensor tasking on a grid.

    The environment represents a grid where a satellite sensor can point to
    different grid positions. The goal is at the center of the grid, and
    rewards are given for reaching positions adjacent to the goal.

    Observation Space:
        Discrete: Grid positions as integers from 0 to (grid_x * grid_y - 1)

    Action Space:
        Discrete(4): [up=0, down=1, left=2, right=3]

    Rewards:
        - 100 for actions from positions adjacent to goal leading to goal
        - 0 for all other actions

    Episode Termination:
        - Goal state is reached (center position)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_x: int = 11,
        grid_y: int = 11,
        render_mode: Optional[str] = None
    ):
        """
        Initialize satellite sensor grid environment.

        Args:
            grid_x: Number of grid cells in x dimension (forced odd)
            grid_y: Number of grid cells in y dimension (forced odd)
            render_mode: Rendering mode for visualization
        """
        super().__init__()

        # CRITICAL: Force odd dimensions (pattern from examples)
        # Reason: Goal must be at exact center
        if grid_x % 2 == 0:
            grid_x += 1
        if grid_y % 2 == 0:
            grid_y += 1

        self.grid_x = grid_x
        self.grid_y = grid_y
        self.num_states = self.grid_x * self.grid_y

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Discrete(self.num_states)

        # Current state
        self.state: Optional[int] = None

        # Goal state at center
        self.goal_state = int(self.num_states / 2)

        # Render mode
        self.render_mode = render_mode

        # Action encoding (from examples)
        self.action_names = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

        # Build reward matrix
        self._build_reward_matrix()

    def _build_reward_matrix(self) -> None:
        """
        Build reward matrix with boundaries and goal rewards.

        Creates a matrix of shape (num_states, 4) where:
        - NaN indicates invalid actions (out of bounds)
        - 100 indicates high reward (adjacent to goal)
        - 0 indicates valid but unrewarded actions
        """
        # Initialize as 3D (grid_x, grid_y, num_actions)
        reward_matrix = np.zeros((self.grid_x, self.grid_y, 4))

        # Set boundaries (NaN for invalid moves)
        # PATTERN: From set_environment_boundary in examples
        reward_matrix[0, :, 0] = np.nan  # No up move from top row
        reward_matrix[-1, :, 1] = np.nan  # No down move from bottom row
        reward_matrix[:, 0, 2] = np.nan  # No left move from left column
        reward_matrix[:, -1, 3] = np.nan  # No right move from right column

        # Set rewards for positions adjacent to goal
        # PATTERN: From set_environment_probabilities in examples
        center_x = int((self.grid_x - 1) / 2)
        center_y = int((self.grid_y - 1) / 2)

        # Position below center gets reward for up action
        reward_matrix[center_x + 1, center_y, 0] = 100
        # Position above center gets reward for down action
        reward_matrix[center_x - 1, center_y, 1] = 100
        # Position to right of center gets reward for left action
        reward_matrix[center_x, center_y + 1, 2] = 100
        # Position to left of center gets reward for right action
        reward_matrix[center_x, center_y - 1, 3] = 100

        # Reshape to (num_states, 4)
        self.reward_matrix = reward_matrix.reshape(self.num_states, 4)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Reset environment to random starting position.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            observation: Initial state (random position)
            info: Additional information (empty dict)
        """
        super().reset(seed=seed)

        # PATTERN: Random start like train() in examples
        # Reason: Explore all possible starting configurations
        self.state = self.np_random.integers(0, self.num_states)

        return self.state, {}

    def step(
        self,
        action: int
    ) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Execute action and return next state, reward, termination.

        Args:
            action: Action to take (0=up, 1=down, 2=left, 3=right)

        Returns:
            observation: Next state
            reward: Reward for transition
            terminated: Whether goal reached
            truncated: Whether episode truncated (always False)
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Check if action is valid (not NaN in reward matrix)
        if np.isnan(self.reward_matrix[self.state, action]):
            # Invalid action (would go out of bounds)
            # Stay in same state, no reward
            reward = 0.0
            terminated = False
        else:
            # Valid action - compute next state
            # PATTERN: From getNextState in examples
            if action == 0:  # up
                next_state = self.state - self.grid_y
            elif action == 1:  # down
                next_state = self.state + self.grid_y
            elif action == 2:  # left
                next_state = self.state - 1
            elif action == 3:  # right
                next_state = self.state + 1
            else:
                raise ValueError(f"Invalid action: {action}")

            # Get reward from reward matrix
            reward = float(self.reward_matrix[self.state, action])

            # Update state
            self.state = next_state

            # Check if goal reached
            terminated = (self.state == self.goal_state)

        truncated = False  # Never truncate (use max_steps in training loop)
        info = {}

        return self.state, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """
        Render current state.

        Returns:
            String representation of state (if render_mode == "human")
        """
        if self.render_mode == "human":
            # Convert state to 2D grid position
            grid_x_pos = self.state // self.grid_y
            grid_y_pos = self.state % self.grid_y

            output = f"State: {self.state} | "
            output += f"Grid Position: ({grid_x_pos}, {grid_y_pos}) | "
            output += f"Goal: {self.goal_state}"

            if self.state == self.goal_state:
                output += " [GOAL REACHED]"

            print(output)
            return output

        return None

    def get_valid_actions(self, state: Optional[int] = None) -> list:
        """
        Get list of valid actions for given state.

        Args:
            state: State to check (uses current state if None)

        Returns:
            List of valid action indices
        """
        if state is None:
            state = self.state

        if state is None:
            raise RuntimeError("State not initialized. Call reset() first.")

        valid_actions = []
        for action in range(4):
            if not np.isnan(self.reward_matrix[state, action]):
                valid_actions.append(action)

        return valid_actions
