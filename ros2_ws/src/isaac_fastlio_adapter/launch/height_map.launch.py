import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    adapter_share = get_package_share_directory("isaac_fastlio_adapter")
    emap_share = get_package_share_directory("elevation_mapping_cupy")
    # Mounted project config; the container sees the repository at /workspace.
    elevation_config = os.environ.get("ELEVATION_CONFIG", "/workspace/config/elevation_isaac.yaml")

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(adapter_share, "launch", "fastlio.launch.py"))
            ),
            Node(
                package="isaac_fastlio_adapter",
                executable="grid_map_to_points.py",
                name="grid_map_to_points",
                output="screen",
                parameters=[{"use_sim_time": True}],
            ),
            Node(
                package="elevation_mapping_cupy",
                executable="elevation_mapping_node.py",
                name="elevation_mapping_node",
                output="screen",
                parameters=[
                    os.path.join(emap_share, "config", "core", "core_param.yaml"),
                    elevation_config,
                    {"use_sim_time": True},
                ],
            ),
        ]
    )
