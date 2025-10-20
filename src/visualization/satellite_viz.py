"""3D visualization functions for satellite gimbal pointing."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Tuple, Optional


def create_satellite_gimbal_visualization(
    satellite_position: np.ndarray,
    target_positions: np.ndarray,
    gimbal_pointing: np.ndarray,
    title: str = "Satellite Gimbal Pointing",
    save_path: Optional[str] = None
) -> None:
    """
    Create 3D visualization of satellite gimbal pointing at ground targets.

    Shows Earth as a blue sphere, satellite position in orbit, ground targets
    on Earth's surface, and gimbal pointing vector from satellite to targets.

    Args:
        satellite_position: [x, y, z] position of satellite in km (ECEF coordinates)
        target_positions: Nx3 array of target positions on ground in km (ECEF)
        gimbal_pointing: [x, y, z] unit vector of gimbal pointing direction
        title: Plot title
        save_path: Optional path to save figure

    Example:
        >>> satellite_pos = np.array([6371 + 500, 0, 0])  # 500 km altitude
        >>> target_pos = np.array([[6371, 0, 0], [6000, 1000, 0]])
        >>> gimbal_vec = np.array([-1, 0, 0])
        >>> create_satellite_gimbal_visualization(satellite_pos, target_pos, gimbal_vec)
    """
    # PATTERN: Use matplotlib 3D for visualization
    # Reason: Lightweight alternative to poliastro for simple gimbal viz
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth (simplified as sphere at origin)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    earth_radius = 6371  # km

    x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(
        x_earth, y_earth, z_earth,
        color='blue',
        alpha=0.3,
        label='Earth'
    )

    # Plot satellite
    ax.scatter(
        satellite_position[0],
        satellite_position[1],
        satellite_position[2],
        color='red',
        s=100,
        label='Satellite',
        marker='^',
        edgecolors='black',
        linewidths=1.5
    )

    # Plot target positions
    if len(target_positions.shape) == 1:
        target_positions = target_positions.reshape(1, -1)

    ax.scatter(
        target_positions[:, 0],
        target_positions[:, 1],
        target_positions[:, 2],
        color='green',
        s=50,
        label='Targets',
        marker='o',
        edgecolors='black',
        linewidths=1
    )

    # Plot gimbal pointing vector
    pointing_length = np.linalg.norm(satellite_position) * 0.5
    pointing_end = satellite_position + gimbal_pointing * pointing_length

    ax.plot(
        [satellite_position[0], pointing_end[0]],
        [satellite_position[1], pointing_end[1]],
        [satellite_position[2], pointing_end[2]],
        color='orange',
        linewidth=3,
        label='Gimbal Pointing'
    )

    ax.set_xlabel('X (km)', fontsize=11)
    ax.set_ylabel('Y (km)', fontsize=11)
    ax.set_zlabel('Z (km)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    # Set equal aspect ratio for 3D plot
    max_range = np.array([
        x_earth.max() - x_earth.min(),
        y_earth.max() - y_earth.min(),
        z_earth.max() - z_earth.min()
    ]).max() / 2.0

    mid_x = (satellite_position[0] + x_earth.mean()) / 2
    mid_y = (satellite_position[1] + y_earth.mean()) / 2
    mid_z = (satellite_position[2] + z_earth.mean()) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Improve viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def grid_to_satellite_coordinates(
    grid_x: int,
    grid_y: int,
    grid_size: int,
    satellite_altitude: float = 500.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert grid position to geographic coordinates and gimbal pointing.

    Maps grid cell indices to latitude/longitude, then to Earth-Centered
    Earth-Fixed (ECEF) coordinates. Computes gimbal pointing vector from
    satellite to target.

    Args:
        grid_x: X position in grid (row index)
        grid_y: Y position in grid (column index)
        grid_size: Total grid dimension (assumes square grid)
        satellite_altitude: Altitude above Earth surface in km (default 500 km)

    Returns:
        Tuple of (target_position, gimbal_vector):
            - target_position: [x, y, z] target position in ECEF coordinates (km)
            - gimbal_vector: [x, y, z] unit vector from satellite to target

    Example:
        >>> target_pos, gimbal_vec = grid_to_satellite_coordinates(11, 11, 23, 500.0)
        >>> target_pos.shape
        (3,)
    """
    # PATTERN: Simple coordinate transform for educational purposes
    # Reason: Map grid positions to lat/lon then to 3D coordinates

    # Map grid indices to lat/lon range
    # Using -45 to 45 degrees for simplicity (covers ~1/4 of Earth)
    lat = (grid_x / grid_size - 0.5) * 90  # -45 to 45 degrees
    lon = (grid_y / grid_size - 0.5) * 90  # -45 to 45 degrees

    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Earth radius (km)
    earth_radius = 6371.0

    # Target position on Earth surface (ECEF coordinates)
    target_x = earth_radius * np.cos(lat_rad) * np.cos(lon_rad)
    target_y = earth_radius * np.cos(lat_rad) * np.sin(lon_rad)
    target_z = earth_radius * np.sin(lat_rad)
    target_position = np.array([target_x, target_y, target_z])

    # Satellite position (directly above center of grid for simplicity)
    # In reality, satellites follow orbits - this is a simplified model
    satellite_position = np.array([earth_radius + satellite_altitude, 0, 0])

    # Gimbal pointing vector (normalized)
    gimbal_vector = target_position - satellite_position
    gimbal_vector = gimbal_vector / np.linalg.norm(gimbal_vector)

    return target_position, gimbal_vector


def create_orbit_visualization(
    satellite_positions: np.ndarray,
    target_position: np.ndarray,
    title: str = "Satellite Orbit",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize satellite orbit around Earth.

    Shows satellite trajectory over time with target position on Earth.

    Args:
        satellite_positions: Nx3 array of satellite positions over time (ECEF, km)
        target_position: [x, y, z] target position on ground (ECEF, km)
        title: Plot title
        save_path: Optional path to save figure

    Example:
        >>> # Create circular orbit
        >>> theta = np.linspace(0, 2*np.pi, 100)
        >>> r = 6371 + 500  # 500 km altitude
        >>> sat_positions = np.column_stack([
        ...     r * np.cos(theta),
        ...     r * np.sin(theta),
        ...     np.zeros(100)
        ... ])
        >>> target = np.array([6371, 0, 0])
        >>> create_orbit_visualization(sat_positions, target)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    earth_radius = 6371  # km

    x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(
        x_earth, y_earth, z_earth,
        color='blue',
        alpha=0.3
    )

    # Plot satellite orbit
    ax.plot(
        satellite_positions[:, 0],
        satellite_positions[:, 1],
        satellite_positions[:, 2],
        color='red',
        linewidth=2,
        label='Satellite Orbit'
    )

    # Plot current satellite position (first point)
    ax.scatter(
        satellite_positions[0, 0],
        satellite_positions[0, 1],
        satellite_positions[0, 2],
        color='red',
        s=100,
        marker='^',
        label='Satellite',
        edgecolors='black',
        linewidths=1.5
    )

    # Plot target
    ax.scatter(
        target_position[0],
        target_position[1],
        target_position[2],
        color='green',
        s=80,
        marker='o',
        label='Target',
        edgecolors='black',
        linewidths=1
    )

    ax.set_xlabel('X (km)', fontsize=11)
    ax.set_ylabel('Y (km)', fontsize=11)
    ax.set_zlabel('Z (km)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    # Set equal aspect ratio
    max_range = np.array([
        x_earth.max() - x_earth.min(),
        y_earth.max() - y_earth.min(),
        z_earth.max() - z_earth.min()
    ]).max() / 2.0

    mid_x = 0
    mid_y = 0
    mid_z = 0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
