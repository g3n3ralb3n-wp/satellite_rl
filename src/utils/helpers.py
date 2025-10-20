"""Helper utilities for satellite RL project."""

import numpy as np
from typing import Tuple


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Sets seed for numpy random number generator to ensure reproducible results.

    Args:
        seed: Random seed value (integer)

    Example:
        >>> set_random_seed(42)
        >>> state = np.random.randint(0, 10)  # Will be reproducible
    """
    np.random.seed(seed)


def grid_index_to_position(state: int, grid_x: int, grid_y: int) -> Tuple[int, int]:
    """
    Convert state index to 2D grid position.

    Args:
        state: State index (0 to grid_x*grid_y - 1)
        grid_x: Grid x dimension
        grid_y: Grid y dimension

    Returns:
        Tuple of (x_position, y_position) in grid

    Example:
        >>> grid_index_to_position(60, 11, 11)
        (5, 5)
    """
    x_pos = state // grid_y
    y_pos = state % grid_y
    return x_pos, y_pos


def grid_position_to_index(x_pos: int, y_pos: int, grid_y: int) -> int:
    """
    Convert 2D grid position to state index.

    Args:
        x_pos: X position in grid
        y_pos: Y position in grid
        grid_y: Grid y dimension

    Returns:
        State index

    Example:
        >>> grid_position_to_index(5, 5, 11)
        60
    """
    return x_pos * grid_y + y_pos


def calculate_convergence_rate(scores: list, window: int = 100) -> float:
    """
    Calculate convergence rate from training scores.

    Measures how quickly the agent is improving by comparing early and late
    performance.

    Args:
        scores: List of episode scores
        window: Window size for averaging

    Returns:
        Convergence rate (higher means faster convergence)

    Example:
        >>> scores = [10, 20, 30, 40, 50]
        >>> calculate_convergence_rate(scores, window=2)
        2.5
    """
    if len(scores) < window * 2:
        return 0.0

    early_avg = np.mean(scores[:window])
    late_avg = np.mean(scores[-window:])

    # Avoid division by zero
    if early_avg == 0:
        return 0.0

    convergence_rate = (late_avg - early_avg) / early_avg

    return convergence_rate


def normalize_scores(scores: list) -> np.ndarray:
    """
    Normalize scores to [0, 1] range.

    Args:
        scores: List of scores to normalize

    Returns:
        Normalized scores as numpy array

    Example:
        >>> normalize_scores([10, 20, 30, 40, 50])
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    scores_array = np.array(scores)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)

    if max_score == min_score:
        return np.zeros_like(scores_array)

    normalized = (scores_array - min_score) / (max_score - min_score)

    return normalized


def latlon_to_ecef(lat: float, lon: float, altitude: float = 0.0) -> np.ndarray:
    """
    Convert latitude/longitude to Earth-Centered Earth-Fixed (ECEF) coordinates.

    Args:
        lat: Latitude in degrees (-90 to 90)
        lon: Longitude in degrees (-180 to 180)
        altitude: Altitude above Earth surface in km (default 0)

    Returns:
        ECEF coordinates [x, y, z] in km

    Example:
        >>> latlon_to_ecef(0, 0, 500)
        array([6871.,    0.,    0.])
    """
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Earth radius (km)
    earth_radius = 6371.0

    # ECEF coordinates
    r = earth_radius + altitude
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return np.array([x, y, z])


def ecef_to_latlon(ecef: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert ECEF coordinates to latitude/longitude/altitude.

    Args:
        ecef: ECEF coordinates [x, y, z] in km

    Returns:
        Tuple of (latitude, longitude, altitude) where:
            - latitude: degrees (-90 to 90)
            - longitude: degrees (-180 to 180)
            - altitude: km above Earth surface

    Example:
        >>> ecef_to_latlon(np.array([6871, 0, 0]))
        (0.0, 0.0, 500.0)
    """
    x, y, z = ecef

    # Earth radius (km)
    earth_radius = 6371.0

    # Calculate altitude
    r = np.sqrt(x**2 + y**2 + z**2)
    altitude = r - earth_radius

    # Calculate latitude
    lat_rad = np.arcsin(z / r)
    lat = np.degrees(lat_rad)

    # Calculate longitude
    lon_rad = np.arctan2(y, x)
    lon = np.degrees(lon_rad)

    return lat, lon, altitude


def compute_pointing_angle(
    satellite_pos: np.ndarray,
    target_pos: np.ndarray
) -> float:
    """
    Compute pointing angle between satellite and target.

    Calculates the angle (in degrees) that the satellite gimbal must rotate
    to point from its current nadir direction to the target.

    Args:
        satellite_pos: Satellite position in ECEF [x, y, z] km
        target_pos: Target position in ECEF [x, y, z] km

    Returns:
        Pointing angle in degrees

    Example:
        >>> sat_pos = np.array([6871, 0, 0])
        >>> target_pos = np.array([6371, 0, 0])
        >>> compute_pointing_angle(sat_pos, target_pos)
        0.0
    """
    # Vector from satellite to target
    pointing_vec = target_pos - satellite_pos

    # Nadir direction (pointing to Earth center from satellite)
    nadir_vec = -satellite_pos

    # Normalize vectors
    pointing_vec_norm = pointing_vec / np.linalg.norm(pointing_vec)
    nadir_vec_norm = nadir_vec / np.linalg.norm(nadir_vec)

    # Compute angle using dot product
    cos_angle = np.dot(pointing_vec_norm, nadir_vec_norm)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg
