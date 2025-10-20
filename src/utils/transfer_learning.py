"""Transfer learning utilities for Q-table resizing."""

import numpy as np
from typing import List, Union


def resize_one_array(array: Union[List[float], np.ndarray], target_size: int) -> np.ndarray:
    """
    Resize 1D array by inserting midpoint values between elements.

    Uses iterative midpoint interpolation to expand array to target size.
    This preserves the value distribution while upscaling, making it suitable
    for transfer learning from smaller to larger grids.

    Algorithm:
        1. Insert midpoints between consecutive elements: [a, b] -> [a, (a+b)/2, b]
        2. Repeat until array length >= target_size
        3. Trim to exact target_size (centered)

    Args:
        array: Input 1D array to resize
        target_size: Desired output size

    Returns:
        Resized array of length target_size

    Example:
        >>> resize_one_array([1.0, 2.0, 3.0], 5)
        array([1. , 1.5, 2. , 2.5, 3. ])
    """
    # PATTERN: Exact implementation from qtable_resize.ipynb
    # CRITICAL: Preserves value distribution while upscaling
    def one_step_expand(val: List[float]) -> List[float]:
        """Insert midpoint between two values."""
        return [val[0], (val[0] + val[1]) / 2, val[1]]

    # Convert to list for manipulation
    if isinstance(array, np.ndarray):
        array = array.tolist()

    n = len(array)

    # Iteratively expand until we reach or exceed target size
    while True:
        array_f = []
        for i in range(n - 1):
            val = one_step_expand([array[i], array[i + 1]])
            if i == n - 2:
                # Last pair - include all three values
                array_f.extend(val)
            else:
                # Intermediate pairs - include first two to avoid duplication
                array_f.extend([val[0], val[1]])

        if len(array_f) >= target_size:
            break

        n = len(array_f)
        array = array_f

    array_f = np.array(array_f)

    # Trim to exact size (centered)
    # If array is larger than target, remove excess from edges equally
    if len(array_f) > target_size:
        offset = int((len(array_f) - target_size) / 2)
        array_f = np.roll(array_f, offset)[-target_size:]

    return array_f


def resize_q_table(q_table: np.ndarray, new_x: int, new_y: int) -> np.ndarray:
    """
    Resize 2D Q-table for transfer learning to larger grid.

    Applies 1D resize operation to both dimensions (rows then columns).
    This allows knowledge learned on a small grid to be transferred to
    a larger grid.

    Args:
        q_table: Original Q-table of shape (grid_x, grid_y)
        new_x: New grid x dimension (rows)
        new_y: New grid y dimension (columns)

    Returns:
        Resized Q-table of shape (new_x, new_y)

    Example:
        >>> q_table = np.random.rand(11, 11)
        >>> resized = resize_q_table(q_table, 23, 23)
        >>> resized.shape
        (23, 23)
    """
    # PATTERN: resize_q_table from qtable_resize.ipynb
    # Resize rows (each row independently)
    shape = q_table.shape
    matrix_new = np.zeros((shape[0], new_x))
    for i in range(shape[0]):
        matrix_new[i] = resize_one_array(q_table[i], new_x)

    # Transpose and resize columns
    matrix = matrix_new.T
    shape = matrix.shape
    matrix_new = np.zeros((shape[0], new_y))
    for i in range(shape[0]):
        matrix_new[i] = resize_one_array(matrix[i], new_y)

    # Transpose back to original orientation
    return matrix_new.T


def q_values_to_q_table(q_matrix: np.ndarray, grid_x: int, grid_y: int) -> np.ndarray:
    """
    Convert Q-matrix (states x actions) to Q-table (value per state).

    Extracts the maximum Q-value for each state and reshapes into a 2D grid.
    This is useful for visualization and transfer learning.

    Args:
        q_matrix: Q-matrix of shape (num_states, num_actions)
        grid_x: Grid x dimension (rows)
        grid_y: Grid y dimension (columns)

    Returns:
        Q-table of shape (grid_x, grid_y) with max Q-value per state

    Raises:
        ValueError: If grid dimensions don't match Q-matrix size
    """
    # PATTERN: create_q_table from examples
    num_states = q_matrix.shape[0]
    expected_states = grid_x * grid_y

    if num_states != expected_states:
        raise ValueError(
            f"Q-matrix has {num_states} states but grid dimensions "
            f"{grid_x}x{grid_y} = {expected_states} states"
        )

    # Extract max Q-value for each state and reshape to grid
    q_table = q_matrix.max(axis=1).reshape(grid_x, grid_y)

    return q_table


def find_highest_neighbor(
    q_table: np.ndarray,
    row: int,
    col: int
) -> tuple:
    """
    Find the highest-valued neighbor of a grid cell.

    Used for reconstructing Q-matrix from Q-table by identifying which
    direction (action) leads to the highest-valued neighboring cell.

    Args:
        q_table: 2D Q-table of state values
        row: Row index of current cell
        col: Column index of current cell

    Returns:
        Tuple of (neighbor_position, direction_name, direction_index):
            - neighbor_position: (row, col) of highest neighbor
            - direction_name: "up", "down", "left", "right", or "goal"
            - direction_index: 0 (up), 1 (down), 2 (left), 3 (right), or -1 (goal)
    """
    # PATTERN: From find_highest_neighbor in examples
    rows, cols = q_table.shape
    highest_neighbor = (row, col)
    max_value = q_table[row, col]
    direction = "goal"
    direction_index = -1

    # Define directions: (row_offset, col_offset, name, index)
    directions = [
        (-1, 0, "up", 0),
        (1, 0, "down", 1),
        (0, -1, "left", 2),
        (0, 1, "right", 3)
    ]

    for dr, dc, dir_name, d_ix in directions:
        r, c = row + dr, col + dc

        # Check if neighbor is within bounds
        if 0 <= r < rows and 0 <= c < cols:
            neighbor_value = q_table[r, c]

            if neighbor_value > max_value:
                max_value = neighbor_value
                highest_neighbor = (r, c)
                direction = dir_name
                direction_index = d_ix

    return highest_neighbor, direction, direction_index


def q_table_to_q_matrix(
    q_table: np.ndarray,
    num_actions: int = 4
) -> np.ndarray:
    """
    Convert Q-table (value per state) to Q-matrix (states x actions).

    Reconstructs a Q-matrix by finding the highest-valued neighbor for each
    cell and assigning that value to the corresponding action. This is an
    approximation used for transfer learning.

    Args:
        q_table: 2D Q-table of state values (grid_x, grid_y)
        num_actions: Number of actions (default 4)

    Returns:
        Q-matrix of shape (num_states, num_actions)

    Note:
        This is an approximation - multiple actions may have zero values
        if they don't lead to the highest neighbor.
    """
    # PATTERN: Adapted from make_q_matrix_from_table in examples
    rows, cols = q_table.shape
    num_states = rows * cols
    q_matrix = np.zeros((num_states, num_actions))

    for i in range(rows):
        for j in range(cols):
            highest_neighbor, _, direction_index = find_highest_neighbor(q_table, i, j)
            a, b = highest_neighbor

            # If a direction was found (not goal), assign value to that action
            if direction_index != -1:
                state_index = i * cols + j
                q_matrix[state_index, direction_index] = q_table[a, b]

    return q_matrix
