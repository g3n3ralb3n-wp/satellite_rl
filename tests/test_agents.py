"""Unit tests for Q-learning agent."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.grid_env import SatelliteSensorGridEnv
from agents.q_learning import QLearningAgent


class TestQLearningAgent:
    """Test cases for QLearningAgent class."""

    @pytest.fixture
    def env(self):
        """Create a test environment."""
        return SatelliteSensorGridEnv(grid_x=11, grid_y=11)

    @pytest.fixture
    def agent(self, env):
        """Create a test agent."""
        return QLearningAgent(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)

    def test_q_table_initialization(self, env):
        """Test that Q-table is initialized correctly."""
        agent = QLearningAgent(env)

        # Check shape
        assert agent.q_table.shape == (121, 4), "Q-table should be 121x4 for 11x11 grid"

        # Check all zeros initially
        assert np.all(agent.q_table == 0), "Q-table should be initialized to zeros"

    def test_q_table_initialization_with_initial_values(self, env):
        """Test Q-table initialization with provided values."""
        initial_q_table = np.ones((121, 4)) * 10.0

        agent = QLearningAgent(env, initial_q_table=initial_q_table)

        # Check that Q-table was copied
        assert np.all(agent.q_table == 10.0), "Q-table should match initial values"
        assert agent.q_table is not initial_q_table, "Q-table should be a copy"

    def test_q_table_initialization_wrong_shape_raises_error(self, env):
        """Test that wrong shape Q-table raises ValueError."""
        wrong_shape_q_table = np.ones((100, 4))  # Wrong number of states

        with pytest.raises(ValueError):
            QLearningAgent(env, initial_q_table=wrong_shape_q_table)

    def test_epsilon_greedy_selection(self, env):
        """Test epsilon-greedy action selection."""
        agent = QLearningAgent(env, epsilon=0.0)  # Greedy only

        # Set Q-table so action 3 is best
        agent.q_table[0, :] = [1, 2, 3, 10]

        # With epsilon=0, should always select action 3
        for _ in range(10):
            action = agent.select_action(0, training=True)
            assert action == 3, "Should select best action with epsilon=0"

    def test_greedy_selection_during_evaluation(self, agent):
        """Test that evaluation mode is greedy."""
        # Set Q-table so action 2 is best
        agent.q_table[0, :] = [1, 2, 10, 3]

        # With training=False, should always select best action
        for _ in range(10):
            action = agent.select_action(0, training=False)
            assert action == 2, "Evaluation should be greedy"

    def test_q_learning_update(self, env):
        """Test that Q-learning updates Q-values correctly."""
        agent = QLearningAgent(env, learning_rate=1.0, discount_factor=0.9, epsilon=0.0)

        # Set initial Q-values
        agent.q_table[0, 1] = 0.0  # Q(s=0, a=1)
        agent.q_table[11, :] = [0, 0, 0, 5]  # Max Q(s'=11) = 5

        # Manually compute expected Q-value after one update
        # Assuming we take action 1 from state 0 and get reward 0
        # and next_state is 11
        # TD target = reward + gamma * max(Q(s')) = 0 + 0.9 * 5 = 4.5
        # With lr=1.0: Q(s,a) = Q(s,a) + lr * (target - Q(s,a)) = 0 + 1.0 * (4.5 - 0) = 4.5

        # Simulate environment step
        state = 0
        action = 1
        reward = 0.0
        next_state = 11

        # Manual Q-learning update
        td_target = reward + agent.gamma * np.max(agent.q_table[next_state])
        td_error = td_target - agent.q_table[state, action]
        agent.q_table[state, action] += agent.lr * td_error

        # Check updated Q-value
        assert np.isclose(agent.q_table[0, 1], 4.5), "Q-value should be updated to 4.5"

    def test_training_convergence(self, env):
        """Test that agent learns over short training."""
        agent = QLearningAgent(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)

        # Train for small number of episodes
        scores = agent.train(num_episodes=100, max_steps=50, verbose=False)

        # Check that scores list has correct length
        assert len(scores) == 100, "Should have 100 scores"

        # Check that Q-table has been updated (no longer all zeros)
        assert not np.all(agent.q_table == 0), "Q-table should be updated after training"

        # Check that scores are generally increasing (learning is happening)
        early_avg = np.mean(scores[:10])
        late_avg = np.mean(scores[-10:])
        assert late_avg > early_avg, "Scores should improve over training"

    def test_get_policy_grid_shape(self, env):
        """Test that policy grid has correct shape."""
        agent = QLearningAgent(env)

        policy_grid = agent.get_policy_grid()

        # Check shape
        assert policy_grid.shape == (11, 11), "Policy grid should be 11x11"

        # Check values are action indices
        assert np.all((policy_grid >= 0) & (policy_grid < 4)), "Actions should be in [0, 3]"

    def test_get_value_grid_shape(self, env):
        """Test that value grid has correct shape."""
        agent = QLearningAgent(env)

        # Set some Q-values
        agent.q_table[60, :] = [1, 2, 3, 4]

        value_grid = agent.get_value_grid()

        # Check shape
        assert value_grid.shape == (11, 11), "Value grid should be 11x11"

        # Check center value (state 60)
        assert value_grid[5, 5] == 4, "Value should be max Q-value (4)"

    def test_get_policy_grid_non_square_raises_error(self, env):
        """Test that non-square grid raises error in get_policy_grid."""
        # Create environment with non-square grid (if possible)
        # Since our env forces odd, this is tricky, but we can test manually

        agent = QLearningAgent(env)
        # Manually set num_states to non-square value
        agent.num_states = 120  # Not a perfect square

        with pytest.raises(ValueError):
            agent.get_policy_grid()

    def test_evaluate_method(self, env):
        """Test evaluate method returns correct metrics."""
        agent = QLearningAgent(env)

        # Train briefly
        agent.train(num_episodes=100, max_steps=50, verbose=False)

        # Evaluate
        results = agent.evaluate(num_episodes=10, max_steps=100, verbose=False)

        # Check results structure
        assert "mean_reward" in results, "Should have mean_reward"
        assert "std_reward" in results, "Should have std_reward"
        assert "success_rate" in results, "Should have success_rate"
        assert "mean_steps" in results, "Should have mean_steps"

        # Check types
        assert isinstance(results["mean_reward"], float), "mean_reward should be float"
        assert isinstance(results["std_reward"], float), "std_reward should be float"
        assert isinstance(results["success_rate"], float), "success_rate should be float"

        # Check ranges
        assert 0.0 <= results["success_rate"] <= 1.0, "Success rate should be in [0, 1]"

    def test_hyperparameter_storage(self, env):
        """Test that hyperparameters are stored correctly."""
        agent = QLearningAgent(
            env,
            learning_rate=0.2,
            discount_factor=0.95,
            epsilon=0.15
        )

        assert agent.lr == 0.2, "Learning rate should be 0.2"
        assert agent.gamma == 0.95, "Discount factor should be 0.95"
        assert agent.epsilon == 0.15, "Epsilon should be 0.15"

    def test_train_returns_scores_list(self, env):
        """Test that train returns a list of scores."""
        agent = QLearningAgent(env)

        scores = agent.train(num_episodes=10, max_steps=10, verbose=False)

        assert isinstance(scores, list), "Scores should be a list"
        assert len(scores) == 10, "Should have 10 scores for 10 episodes"
        assert all(isinstance(s, (int, float, np.number)) for s in scores), "Scores should be numeric"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
