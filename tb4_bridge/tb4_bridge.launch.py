
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    robot_ns_arg = DeclareLaunchArgument(
        "robot_ns",
        default_value="/Turtlebot_02493",
        description="TurtleBot4 namespace on the network",
    )

    domain_id_arg = DeclareLaunchArgument(
        "domain_id",
        default_value="3",
        description="ROS_DOMAIN_ID for the TurtleBot4",
    )

    set_domain_id = SetEnvironmentVariable(
        name="ROS_DOMAIN_ID",
        value=LaunchConfiguration("domain_id"),
    )

    bridge_node = Node(
        package=None,  
        executable=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tb4_bridge_node.py"
        ),
        name="tb4_bridge",
        output="screen",
        emulate_tty=True,
    )


    return LaunchDescription([
        robot_ns_arg,
        domain_id_arg,
        set_domain_id,
        LogInfo(msg=["Launching TB4 bridge for namespace: ", LaunchConfiguration("robot_ns")]),
        LogInfo(msg=["ROS_DOMAIN_ID: ", LaunchConfiguration("domain_id")]),
        bridge_node,
    ])
