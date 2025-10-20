"""Q-learning agent for satellite sensor tasking."""

import numpy as np
from typing import Optional, List
import gymnasium as gym


class QLearningAgent:
    """
    Q-learning agent for satellite sensor tasking.

    Implements tabular Q-learning with epsilon-greedy exploration.
    Supports transfer learning via Q-table initialization.

    The Q-table stores expected cumulative rewards for taking each action
    from each state. The agent learns by updating Q-values based on
    observed rewards and estimated future rewards.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        initial_q_table: Optional[np.ndarray] = None
    ):
        """
        Initialize Q-learning agent.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate (alpha) - controls how much new
                information overrides old
            discount_factor: Discount factor (gamma) - importance of future
                rewards (0-1)
            epsilon: Exploration rate for epsilon-greedy policy (0-1)
            initial_q_table: Optional pre-trained Q-table for transfer learning
                Should have shape (num_states, num_actions)
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Get environment dimensions
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n

        # Initialize Q-table
        if initial_q_table is not None:
            if initial_q_table.shape != (self.num_states, self.num_actions):
                raise ValueError(
                    f"Initial Q-table shape {initial_q_table.shape} does not "
                    f"match environment ({self.num_states}, {self.num_actions})"
                )
            self.q_table = initial_q_table.copy()
            print("Using provided Q-table for warm start.")
        else:
            self.q_table = np.zeros((self.num_states, self.num_actions))

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        During training, explores with probability epsilon and exploits
        with probability (1-epsilon). During evaluation, always exploits.

        Args:
            state: Current state
            training: If True, use epsilon-greedy; else use greedy

        Returns:
            Selected action index
        """
        # PATTERN: Epsilon-greedy from train() in examples
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return self.env.action_space.sample()
        else:
            # Exploit: best action from Q-table
            return int(np.argmax(self.q_table[state]))

    def train(
        self,
        num_episodes: int = 30000,
        max_steps: int = 100,
        verbose: bool = True
    ) -> List[float]:
        """
        Train agent using Q-learning algorithm.

        Updates Q-table using the Q-learning update rule:
        Q(s,a) = Q(s,a) + lr * [r + gamma * max(Q(s',a')) - Q(s,a)]

        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Print progress every 1000 episodes

        Returns:
            List of episode scores (sum of Q-table values) for tracking
            convergence
        """
        scores = []

        for episode in range(num_episodes):
            # Reset environment
            state, _ = self.env.reset()
            episode_score = 0

            for step in range(max_steps):
                # Select action using epsilon-greedy
                action = self.select_action(state, training=True)

                # Take step in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Q-learning update
                # PATTERN: Exact formula from train() in examples
                # TD target: r + gamma * max(Q(s', a'))
                td_target = reward + self.gamma * np.max(self.q_table[next_state])
                # TD error: target - current estimate
                td_error = td_target - self.q_table[state, action]
                # Update Q-value
                self.q_table[state, action] += self.lr * td_error

                episode_score += reward
                state = next_state

                if terminated or truncated:
                    break

            # Track convergence score (sum of all Q-values)
            # PATTERN: From game_score in examples
            convergence_score = np.sum(self.q_table)
            scores.append(convergence_score)

            if verbose and episode % 1000 == 0:
                avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                print(f"Episode {episode}/{num_episodes}, Avg Score: {avg_score:.2f}")

        if verbose:
            print("Training done...")

        return scores

    def get_policy_grid(self) -> np.ndarray:
        """
        Get policy as 2D grid showing best action at each state.

        Returns:
            2D array of action indices where policy_grid[i,j] is the best
            action to take from grid position (i,j)
        """
        # PATTERN: Similar to get_q_table_indexes from examples
        # Compute grid size (assumes square grid)
        grid_size = int(np.sqrt(self.num_states))

        if grid_size * grid_size != self.num_states:
            raise ValueError(
                f"Cannot reshape {self.num_states} states into square grid. "
                f"Grid must be square (e.g., 11x11, 23x23)"
            )

        # Get best action for each state
        policy = np.argmax(self.q_table, axis=1).reshape(grid_size, grid_size)

        return policy

    def get_value_grid(self) -> np.ndarray:
        """
        Get state values as 2D grid for visualization.

        State value V(s) = max_a Q(s,a) - the expected cumulative reward
        from state s when following the optimal policy.

        Returns:
            2D array of state values where value_grid[i,j] is the value
            of grid position (i,j)
        """
        # PATTERN: create_q_table from examples
        # Compute grid size (assumes square grid)
        grid_size = int(np.sqrt(self.num_states))

        if grid_size * grid_size != self.num_states:
            raise ValueError(
                f"Cannot reshape {self.num_states} states into square grid. "
                f"Grid must be square (e.g., 11x11, 23x23)"
            )

        # Get max Q-value for each state
        values = np.max(self.q_table, axis=1).reshape(grid_size, grid_size)

        return values

    def evaluate(
        self,
        num_episodes: int = 100,
        max_steps: int = 100,
        verbose: bool = False
    ) -> dict:
        """
        Evaluate agent performance (no learning).

        Args:
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            verbose: Print episode results

        Returns:
            Dictionary with evaluation metrics:
                - mean_reward: Average total reward per episode
                - std_reward: Standard deviation of rewards
                - success_rate: Fraction of episodes reaching goal
                - mean_steps: Average steps to reach goal (successful episodes)
        """
        total_rewards = []
        successes = 0
        steps_to_goal = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0

            for step in range(max_steps):
                # Use greedy policy (no exploration)
                action = self.select_action(state, training=False)

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward
                episode_steps += 1
                state = next_state

                if terminated:
                    successes += 1
                    steps_to_goal.append(episode_steps)
                    break

                if truncated:
                    break

            total_rewards.append(episode_reward)

            if verbose:
                status = "SUCCESS" if terminated else "FAILED"
                print(f"Episode {episode + 1}: Reward={episode_reward:.1f}, Steps={episode_steps}, {status}")

        results = {
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "success_rate": successes / num_episodes,
            "mean_steps": float(np.mean(steps_to_goal)) if steps_to_goal else None
        }

        return results
