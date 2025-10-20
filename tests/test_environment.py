"""Unit tests for satellite sensor grid environment."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.grid_env import SatelliteSensorGridEnv


class TestSatelliteSensorGridEnv:
    """Test cases for SatelliteSensorGridEnv class."""

    def test_grid_dimensions_forced_odd(self):
        """Test that even grid dimensions are forced to odd."""
        # Test even x dimension
        env = SatelliteSensorGridEnv(grid_x=10, grid_y=11)
        assert env.grid_x == 11, "Even grid_x should be forced to 11"
        assert env.grid_y == 11, "Odd grid_y should remain 11"

        # Test even y dimension
        env = SatelliteSensorGridEnv(grid_x=11, grid_y=10)
        assert env.grid_x == 11, "Odd grid_x should remain 11"
        assert env.grid_y == 11, "Even grid_y should be forced to 11"

        # Test both even
        env = SatelliteSensorGridEnv(grid_x=10, grid_y=10)
        assert env.grid_x == 11, "Even grid_x should be forced to 11"
        assert env.grid_y == 11, "Even grid_y should be forced to 11"

        # Test both odd
        env = SatelliteSensorGridEnv(grid_x=11, grid_y=11)
        assert env.grid_x == 11, "Odd grid_x should remain 11"
        assert env.grid_y == 11, "Odd grid_y should remain 11"

    def test_goal_state_is_center(self):
        """Test that goal state is at center position."""
        # Test 11x11 grid
        env = SatelliteSensorGridEnv(grid_x=11, grid_y=11)
        expected_goal = int((11 * 11) / 2)  # 60
        assert env.goal_state == expected_goal, f"Goal should be at {expected_goal}"

        # Test 23x23 grid
        env = SatelliteSensorGridEnv(grid_x=23, grid_y=23)
        expected_goal = int((23 * 23) / 2)  # 264
        assert env.goal_state == expected_goal, f"Goal should be at {expected_goal}"

    def test_observation_action_spaces(self):
        """Test that observation and action spaces are correct."""
        env = SatelliteSensorGridEnv(grid_x=11, grid_y=11)

        # Check observation space
        assert env.observation_space.n == 121, "11x11 grid should have 121 states"

        # Check action space
        assert env.action_space.n == 4, "Should have 4 actions (up, down, left, right)"

    def test_reset_returns_valid_state(self):
        """Test that reset returns a valid state."""
        env = SatelliteSensorGridEnv(grid_x=11, grid_y=11)

        # Test multiple resets
        for _ in range(10):
            state, info = env.reset(seed=None)

            # Check state is valid
            assert isinstance(state, (int, np.integer)), "State should be integer"
            assert 0 <= state < 121, "State should be in valid range [0, 121)"

            # Check info is dict
            assert isinstance(info, dict), "Info should be dictionary"

    def test_reset_with_seed_is_reproducible(self):
        """Test that reset with seed produces reproducible results."""
        env = SatelliteSensorGridEnv(grid_x=11, grid_y=11)

        # Reset with same seed multiple times
        state1, _ = env.reset(seed=42)
        state2, _ = env.reset(seed=42)

        assert state1 == state2, "Same seed should produce same initial state"

    def test_step_respects_boundaries(self):
        """Test that step correctly handles boundary conditions."""
        env = SatelliteSensorGridEnv(grid_x=11, grid_y=11)

        # Test top-left corner (state 0)
        env.state = 0

        # Try to go up (should stay at 0, no reward)
        next_state, reward, terminated, truncated, _ = env.step(0)  # up
        assert env.state == 0, "Should stay at boundary"
        assert reward == 0.0, "No reward for invalid move"
        assert not terminated, "Should not terminate"

        # Try to go left (should stay at 0, no reward)
        next_state, reward, terminated, truncated, _ = env.step(2)  # left
        assert env.state == 0, "Should stay at boundary"
        assert reward == 0.0, "No reward for invalid move"

    def test_rewards_adjacent_to_goal(self):
        """Test that rewards are correctly placed adjacent to goal."""
        env = SatelliteSensorGridEnv(grid_x=11, grid_y=11)

        # Goal is at state 60 (center of 11x11)
        goal_state = 60

        # Test position below goal (state 71) going up
        env.state = goal_state + 11  # State 71
        next_state, reward, terminated, truncated, _ = env.step(0)  # up
        assert reward == 100.0, "Should get reward 100 for move to goal from below"
        assert terminated, "Should terminate when goal reached"

        # Reset and test position above goal (state 49) going down
        env.reset(seed=42)
        env.state = goal_state - 11  # State 49
        next_state, reward, terminated, truncated, _ = env.step(1)  # down
        assert reward == 100.0, "Should get reward 100 for move to goal from above"
        assert terminated, "Should terminate when goal reached"

        # Reset and test position to right of goal (state 61) going left
        env.reset(seed=42)
        env.state = goal_state + 1  # State 61
        next_state, reward, terminated, truncated, _ = env.step(2)  # left
        assert reward == 100.0, "Should get reward 100 for move to goal from right"
        assert terminated, "Should terminate when goal reached"

        # Reset and test position to left of goal (state 59) going right
        env.reset(seed=42)
        env.state = goal_state - 1  # State 59
        next_state, reward, terminated, truncated, _ = env.step(3)  # right
        assert reward == 100.0, "Should get reward 100 for move to goal from left"
        assert terminated, "Should terminate when goal reached"

    def test_get_valid_actions(self):
        """Test get_valid_actions method."""
        env = SatelliteSensorGridEnv(grid_x=11, grid_y=11)

        # Test corner (top-left, state 0)
        env.state = 0
        valid_actions = env.get_valid_actions()
        assert 0 not in valid_actions, "Cannot go up from top row"
        assert 2 not in valid_actions, "Cannot go left from left column"
        assert 1 in valid_actions, "Can go down from top row"
        assert 3 in valid_actions, "Can go right from left column"

        # Test center (state 60)
        env.state = 60
        valid_actions = env.get_valid_actions()
        assert len(valid_actions) == 4, "All actions valid from center"

    def test_action_mechanics(self):
        """Test that actions correctly transition between states."""
        env = SatelliteSensorGridEnv(grid_x=11, grid_y=11)

        # Start at state 60 (center)
        env.state = 60

        # Test up action (0): should subtract grid_y
        initial_state = env.state
        env.step(0)  # up
        assert env.state == initial_state - 11, "Up should subtract 11"

        # Reset to center
        env.state = 60

        # Test down action (1): should add grid_y
        env.step(1)  # down
        assert env.state == 60 + 11, "Down should add 11"

        # Reset to center
        env.state = 60

        # Test left action (2): should subtract 1
        env.step(2)  # left
        assert env.state == 59, "Left should subtract 1"

        # Reset to center
        env.state = 60

        # Test right action (3): should add 1
        env.step(3)  # right
        assert env.state == 61, "Right should add 1"

    def test_render_human_mode(self):
        """Test render in human mode."""
        env = SatelliteSensorGridEnv(grid_x=11, grid_y=11, render_mode="human")
        env.reset(seed=42)

        output = env.render()
        assert output is not None, "Render should return string in human mode"
        assert "State:" in output, "Output should contain state info"
        assert "Grid Position:" in output, "Output should contain grid position"

    def test_step_returns_correct_tuple_format(self):
        """Test that step returns correct tuple format (gymnasium API)."""
        env = SatelliteSensorGridEnv(grid_x=11, grid_y=11)
        env.reset(seed=42)

        result = env.step(1)  # Take a step

        # Should return 5-tuple
        assert len(result) == 5, "Step should return 5-tuple"

        observation, reward, terminated, truncated, info = result

        # Check types
        assert isinstance(observation, (int, np.integer)), "Observation should be int"
        assert isinstance(reward, float), "Reward should be float"
        assert isinstance(terminated, bool), "Terminated should be bool"
        assert isinstance(truncated, bool), "Truncated should be bool"
        assert isinstance(info, dict), "Info should be dict"

    def test_num_states_calculation(self):
        """Test that num_states is calculated correctly."""
        env = SatelliteSensorGridEnv(grid_x=11, grid_y=11)
        assert env.num_states == 121, "11x11 should have 121 states"

        env = SatelliteSensorGridEnv(grid_x=23, grid_y=23)
        assert env.num_states == 529, "23x23 should have 529 states"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
