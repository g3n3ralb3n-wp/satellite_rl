"""Unit tests for transfer learning utilities."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.transfer_learning import (
    resize_one_array,
    resize_q_table,
    q_values_to_q_table,
    find_highest_neighbor,
    q_table_to_q_matrix
)


class TestTransferLearningUtilities:
    """Test cases for transfer learning utilities."""

    def test_resize_one_array_output_size(self):
        """Test that resize_one_array produces correct output size."""
        array = [1.0, 2.0, 3.0]

        # Resize to 5
        resized = resize_one_array(array, 5)
        assert len(resized) == 5, "Should resize to exactly 5 elements"

        # Resize to 7
        resized = resize_one_array(array, 7)
        assert len(resized) == 7, "Should resize to exactly 7 elements"

    def test_resize_one_array_preserves_endpoints(self):
        """Test that resize_one_array preserves array endpoints."""
        array = [1.0, 5.0, 10.0]

        resized = resize_one_array(array, 7)

        # Check endpoints are preserved
        assert np.isclose(resized[0], 1.0), "First element should be preserved"
        assert np.isclose(resized[-1], 10.0), "Last element should be preserved"

    def test_resize_one_array_interpolation(self):
        """Test that resize_one_array uses midpoint interpolation."""
        array = [0.0, 10.0]

        resized = resize_one_array(array, 3)

        # Should insert midpoint: [0, 5, 10]
        assert np.isclose(resized[1], 5.0), "Should insert midpoint value"

    def test_resize_one_array_accepts_numpy_array(self):
        """Test that resize_one_array accepts numpy arrays."""
        array = np.array([1.0, 2.0, 3.0])

        resized = resize_one_array(array, 5)

        assert len(resized) == 5, "Should work with numpy array input"
        assert isinstance(resized, np.ndarray), "Should return numpy array"

    def test_resize_q_table_dimensions(self):
        """Test that resize_q_table produces correct dimensions."""
        q_table = np.random.rand(11, 11)

        resized = resize_q_table(q_table, 23, 23)

        assert resized.shape == (23, 23), "Should resize to 23x23"

    def test_resize_q_table_smaller_to_larger(self):
        """Test resizing from smaller to larger grid."""
        # Create 3x3 Q-table
        q_table = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)

        # Resize to 5x5
        resized = resize_q_table(q_table, 5, 5)

        assert resized.shape == (5, 5), "Should be 5x5"

        # Check that corner values are preserved (approximately)
        assert np.isclose(resized[0, 0], 1, atol=0.5), "Top-left should be close to original"
        assert np.isclose(resized[-1, -1], 9, atol=0.5), "Bottom-right should be close to original"

    def test_resize_q_table_preserves_center_values(self):
        """Test that resizing preserves approximate center values."""
        # Create 11x11 Q-table with high value at center
        q_table = np.zeros((11, 11))
        q_table[5, 5] = 100.0  # Center value

        # Resize to 23x23
        resized = resize_q_table(q_table, 23, 23)

        # Center of 23x23 is at (11, 11)
        # Should have high value near center
        center_region = resized[10:13, 10:13]
        assert np.max(center_region) > 50, "Center region should have high values"

    def test_q_values_to_q_table_shape(self):
        """Test that q_values_to_q_table produces correct shape."""
        q_matrix = np.random.rand(121, 4)  # 11x11 grid

        q_table = q_values_to_q_table(q_matrix, 11, 11)

        assert q_table.shape == (11, 11), "Should be 11x11"

    def test_q_values_to_q_table_max_extraction(self):
        """Test that q_values_to_q_table extracts max values."""
        # Create Q-matrix with known max values
        q_matrix = np.zeros((9, 4))  # 3x3 grid
        q_matrix[0, :] = [1, 2, 3, 10]  # Max is 10
        q_matrix[4, :] = [20, 5, 3, 1]  # Max is 20

        q_table = q_values_to_q_table(q_matrix, 3, 3)

        assert q_table[0, 0] == 10, "Should extract max value (10)"
        assert q_table[1, 1] == 20, "Should extract max value (20)"

    def test_q_values_to_q_table_wrong_dimensions_raises_error(self):
        """Test that wrong dimensions raise ValueError."""
        q_matrix = np.random.rand(100, 4)  # 100 states

        with pytest.raises(ValueError):
            q_values_to_q_table(q_matrix, 11, 11)  # 11x11 = 121 states (mismatch)

    def test_find_highest_neighbor_returns_correct_format(self):
        """Test that find_highest_neighbor returns correct tuple format."""
        q_table = np.random.rand(5, 5)

        neighbor, direction, index = find_highest_neighbor(q_table, 2, 2)

        # Check types
        assert isinstance(neighbor, tuple), "Neighbor should be tuple"
        assert len(neighbor) == 2, "Neighbor should have 2 elements"
        assert isinstance(direction, str), "Direction should be string"
        assert isinstance(index, int), "Index should be int"

    def test_find_highest_neighbor_at_max(self):
        """Test finding highest neighbor when current cell is max."""
        q_table = np.array([
            [1, 2, 3],
            [4, 10, 6],
            [7, 8, 9]
        ], dtype=float)

        # Center cell (1,1) has value 10 (highest)
        neighbor, direction, index = find_highest_neighbor(q_table, 1, 1)

        # Should return self as neighbor
        assert neighbor == (1, 1), "Should return self when current is highest"
        assert direction == "goal", "Direction should be 'goal'"
        assert index == -1, "Index should be -1"

    def test_find_highest_neighbor_finds_correct_direction(self):
        """Test that highest neighbor is found in correct direction."""
        q_table = np.array([
            [1, 2, 3],
            [4, 5, 20],  # Highest is to the right
            [7, 8, 9]
        ], dtype=float)

        # Check cell (1, 1)
        neighbor, direction, index = find_highest_neighbor(q_table, 1, 1)

        assert neighbor == (1, 2), "Should find neighbor at (1, 2)"
        assert direction == "right", "Direction should be 'right'"
        assert index == 3, "Index should be 3 (right)"

    def test_q_table_to_q_matrix_shape(self):
        """Test that q_table_to_q_matrix produces correct shape."""
        q_table = np.random.rand(11, 11)

        q_matrix = q_table_to_q_matrix(q_table, num_actions=4)

        assert q_matrix.shape == (121, 4), "Should be 121x4"

    def test_q_table_to_q_matrix_assigns_values(self):
        """Test that q_table_to_q_matrix assigns values to correct actions."""
        q_table = np.array([
            [1, 2, 3],
            [4, 10, 6],  # Center is max
            [7, 8, 9]
        ], dtype=float)

        q_matrix = q_table_to_q_matrix(q_table, num_actions=4)

        # Check that values are assigned based on highest neighbors
        # Cell (0, 0) has neighbor (1, 1) with value 10 in down direction
        state_0 = 0 * 3 + 0  # State index 0
        # Action 1 (down) should have value from neighbor
        assert q_matrix[state_0, 1] > 0, "Should have non-zero value for best direction"

    def test_resize_chain(self):
        """Test full resize chain: Q-matrix -> Q-table -> resize -> Q-matrix."""
        # Create small Q-matrix (3x3 grid)
        small_q_matrix = np.random.rand(9, 4)

        # Convert to Q-table
        small_q_table = q_values_to_q_table(small_q_matrix, 3, 3)

        # Resize to larger grid
        large_q_table = resize_q_table(small_q_table, 5, 5)

        # Convert back to Q-matrix
        large_q_matrix = q_table_to_q_matrix(large_q_table, num_actions=4)

        # Check final shape
        assert large_q_matrix.shape == (25, 4), "Final Q-matrix should be 25x4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
