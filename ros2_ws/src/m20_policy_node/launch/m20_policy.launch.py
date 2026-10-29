from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("policy_path", description="TorchScript policy.pt exported by isaaclab play"),
            DeclareLaunchArgument("device", default_value="cpu"),
            DeclareLaunchArgument("use_sim_time", default_value="true", description="Follow /clock; set false on the real robot"),
            Node(
                package="m20_policy_node",
                executable="m20_policy_node",
                name="m20_policy_node",
                output="screen",
                parameters=[
                    {
                        "policy_path": LaunchConfiguration("policy_path"),
                        "device": LaunchConfiguration("device"),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    }
                ],
            ),
        ]
    )
