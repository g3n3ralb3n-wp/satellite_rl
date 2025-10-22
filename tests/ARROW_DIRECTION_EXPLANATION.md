# Arrow Direction Issue - Complete Explanation

## TL;DR: The visualization code IS CORRECT! The problem is sparse rewards make training very slow.

## The Core Issue

Your visualization shows arrows pointing the wrong direction **not because the code is wrong**, but because **the agent hasn't fully learned the correct policy yet**, even after 600,000 episodes!

## Why Is Training So Slow?

The environment only gives rewards at 4 positions (directly adjacent to the goal):
- Position (4,5): reward +100 for action "down"
- Position (6,5): reward +100 for action "up"
- Position (5,4): reward +100 for action "right"
- Position (5,6): reward +100 for action "left"

All other 117 positions in the 11×11 grid (121 total states) must learn through **value propagation**:

1. Agent randomly explores and eventually learns positions adjacent to goal get rewards
2. Values slowly propagate backward through the Bellman equation
3. Position (0,5) is 5 steps from the goal → needs values to propagate through (4,5) → (3,5) → (2,5) → (1,5) → (0,5)
4. Each propagation step requires thousands of episodes with proper exploration

## The Visualization Code is CORRECT

Current settings in `grid_viz.py`:
```python
arrow_dx = {0: 0, 1: 0, 2: -0.3, 3: 0.3}
arrow_dy = {0: -0.3, 1: 0.3, 2: 0, 3: 0}
```

With `origin='upper'`:
- Row 0 is at TOP of plot (y=0)
- Row 10 is at BOTTOM of plot (y=10)
- Y-axis increases downward
- Matplotlib arrow with `+dy` points in direction of increasing y → points DOWNWARD
- Matplotlib arrow with `-dy` points in direction of decreasing y → points UPWARD

Therefore:
- **Action 0 (up)**: `arrow_dy = -0.3` → arrow points UPWARD ✓
- **Action 1 (down)**: `arrow_dy = +0.3` → arrow points DOWNWARD ✓
- **Action 2 (left)**: `arrow_dx = -0.3` → arrow points LEFT ✓
- **Action 3 (right)**: `arrow_dx = +0.3` → arrow points RIGHT ✓

## Proof: Test Images

See `tests/correct_policy_simple.png` and `tests/correct_policy_annotated.png`

These show a **properly trained** 5×5 agent where:
- All arrows point toward the goal at (2,2)
- Top positions: arrows point DOWN ↓
- Bottom positions: arrows point UP ↑
- Left positions: arrows point RIGHT →
- Right positions: arrows point LEFT ←

This proves the visualization code works correctly!

## Solution: Better Training Parameters

For 11×11 grid, use:

```python
agent = QLearningAgent(
    env=env,
    learning_rate=0.15,      # Higher for faster learning
    discount_factor=0.95,    # Higher for better value propagation
    epsilon=0.02             # Low for deterministic policy
)

agent.train(num_episodes=300000, max_steps=50, verbose=True)
```

Even better: 500,000+ episodes for perfect convergence at all edge positions.

## Verification

After training, check middle column (col 5):
- Rows 0-4 should learn action 1 (down) to move toward goal at row 5
- Rows 6-10 should learn action 0 (up) to move toward goal at row 5

You'll notice rows 6-10 learn correctly first (bottom half), while rows 0-4 (top half) take much longer. This is because of how values propagate upward through the Q-table.

## Alternative: Different Reward Structure

If you want faster learning, you could modify the environment to give distance-based rewards:
```python
reward = 100 - manhattan_distance(current_pos, goal_pos)
```

But this changes the problem and isn't how the original examples work.

## Bottom Line

**Your arrows will point correctly once the agent fully trains!**

The test images in `tests/` show what correct arrows look like. The visualization code is working as designed.
