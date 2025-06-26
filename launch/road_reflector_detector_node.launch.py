import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    share_dir = get_package_share_directory('road_reflector_detector')
    param_file = os.path.join(share_dir, 'config', 'params.yaml')

    container = ComposableNodeContainer(
        name='road_reflector_detector_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='road_reflector_detector',
                plugin='road_reflector_detector::RoadReflectorDetectorNode',
                name='road_reflector_detector_node',
                parameters=[param_file],
                extra_arguments=[{'use_intra_process_comms': True}]
            )
        ],
        output='screen',
    )

    return LaunchDescription([container])