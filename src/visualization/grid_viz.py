"""Visualization functions for 2D grid-based RL."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional


def plot_learning_curve(
    scores: List[float],
    window: int = 100,
    title: str = "Learning Curve",
    save_path: Optional[str] = None
) -> None:
    """
    Plot learning curve with moving average.

    Visualizes agent learning progress over episodes. Raw scores are plotted
    with low opacity, and a moving average smooths the curve to show trends.

    Args:
        scores: List of episode scores (typically cumulative Q-values or rewards)
        window: Window size for moving average smoothing
        title: Plot title
        save_path: Optional path to save figure (e.g., "learning_curve.png")

    Example:
        >>> scores = agent.train(num_episodes=10000)
        >>> plot_learning_curve(scores, window=500)
    """
    # PATTERN: Enhanced version of q_scores plotting from examples
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot raw scores (faint)
    ax.plot(scores, alpha=0.3, color='blue', label='Episode Score', linewidth=0.5)

    # Plot moving average
    if len(scores) >= window:
        # Compute moving average using convolution
        moving_avg = np.convolve(scores, np.ones(window) / window, mode='valid')
        ax.plot(
            range(window - 1, len(scores)),
            moving_avg,
            color='red',
            linewidth=2,
            label=f'{window}-Episode Moving Avg'
        )

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_value_heatmap(
    value_grid: np.ndarray,
    title: str = "State Value Function",
    cmap: str = 'jet',
    save_path: Optional[str] = None
) -> None:
    """
    Plot state values as heatmap.

    Visualizes the learned value function V(s) = max_a Q(s,a) as a heatmap.
    Brighter/warmer colors indicate higher values (better states).

    Args:
        value_grid: 2D array of state values (grid_x, grid_y)
        title: Plot title
        cmap: Colormap name (e.g., 'jet', 'viridis', 'hot')
        save_path: Optional path to save figure

    Example:
        >>> value_grid = agent.get_value_grid()
        >>> plot_value_heatmap(value_grid, title="Learned Values (11x11)")
    """
    # PATTERN: Enhanced plt.imshow from examples with colorbar
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create heatmap
    im = ax.imshow(value_grid, cmap=cmap, origin='lower', aspect='auto')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('State Value', rotation=270, labelpad=20, fontsize=11)

    # Add grid lines for clarity
    ax.set_xticks(np.arange(value_grid.shape[1]))
    ax.set_yticks(np.arange(value_grid.shape[0]))
    ax.grid(True, color='white', linewidth=0.5, alpha=0.5)

    # Add value annotations for small grids
    if value_grid.shape[0] <= 15 and value_grid.shape[1] <= 15:
        for i in range(value_grid.shape[0]):
            for j in range(value_grid.shape[1]):
                text = ax.text(
                    j, i, f'{value_grid[i, j]:.1f}',
                    ha="center", va="center", color="black", fontsize=6
                )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_policy_arrows(
    policy_grid: np.ndarray,
    value_grid: Optional[np.ndarray] = None,
    title: str = "Learned Policy",
    save_path: Optional[str] = None
) -> None:
    """
    Plot policy as arrows on grid, optionally with value heatmap background.

    Visualizes the learned policy π(s) = argmax_a Q(s,a) as arrows showing
    the best action to take from each state.

    Action encoding:
        - 0: up (↑)
        - 1: down (↓)
        - 2: left (←)
        - 3: right (→)

    Args:
        policy_grid: 2D array of action indices (grid_x, grid_y)
        value_grid: Optional 2D array of state values for background heatmap
        title: Plot title
        save_path: Optional path to save figure

    Example:
        >>> policy_grid = agent.get_policy_grid()
        >>> value_grid = agent.get_value_grid()
        >>> plot_policy_arrows(policy_grid, value_grid)
    """
    # PATTERN: Enhanced visualization combining direction_encodings with heatmap
    fig, ax = plt.subplots(figsize=(10, 10))

    # Background heatmap if provided
    if value_grid is not None:
        im = ax.imshow(
            value_grid,
            cmap='gray',
            origin='lower',
            alpha=0.5,
            aspect='auto'
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('State Value', rotation=270, labelpad=15, fontsize=10)

    # Arrow directions (dx, dy for each action)
    # Actions: 0=up, 1=down, 2=left, 3=right
    arrow_dx = {0: 0, 1: 0, 2: -0.3, 3: 0.3}
    arrow_dy = {0: 0.3, 1: -0.3, 2: 0, 3: 0}

    # Plot arrows for each grid cell
    for i in range(policy_grid.shape[0]):
        for j in range(policy_grid.shape[1]):
            action = policy_grid[i, j]
            ax.arrow(
                j, i,  # Start position (col, row)
                arrow_dx[action],
                arrow_dy[action],
                head_width=0.2,
                head_length=0.2,
                fc='red',
                ec='red',
                alpha=0.8,
                linewidth=1.5
            )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_xticks(np.arange(policy_grid.shape[1]))
    ax.set_yticks(np.arange(policy_grid.shape[0]))
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set axis limits to show full grid
    ax.set_xlim(-0.5, policy_grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, policy_grid.shape[0] - 0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def compare_learning_curves(
    scores_dict: dict,
    window: int = 100,
    title: str = "Learning Curve Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple learning curves for comparison.

    Useful for comparing different algorithms, hyperparameters, or
    transfer learning vs. from-scratch training.

    Args:
        scores_dict: Dictionary mapping labels to score lists
            e.g., {"Transfer Learning": scores1, "From Scratch": scores2}
        window: Window size for moving average
        title: Plot title
        save_path: Optional path to save figure

    Example:
        >>> scores_transfer = agent_transfer.train(num_episodes=10000)
        >>> scores_scratch = agent_scratch.train(num_episodes=10000)
        >>> compare_learning_curves({
        ...     "Transfer Learning": scores_transfer,
        ...     "From Scratch": scores_scratch
        ... })
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for idx, (label, scores) in enumerate(scores_dict.items()):
        color = colors[idx % len(colors)]

        # Plot moving average
        if len(scores) >= window:
            moving_avg = np.convolve(scores, np.ones(window) / window, mode='valid')
            ax.plot(
                range(window - 1, len(scores)),
                moving_avg,
                color=color,
                linewidth=2,
                label=label
            )

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score (Moving Average)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
