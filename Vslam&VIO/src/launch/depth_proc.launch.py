from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_dir = get_package_share_directory('my_depth_proc')
    config_file = os.path.join(package_dir, 'config', 'depth_proc_params.yaml')
    
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/camera/depth/image_raw/compressed',
        description='Input compressed depth topic'
    )
    
    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/camera/depth/image',
        description='Output depth topic'
    )
    
    depth_converter_node = Node(
        package='my_depth_proc',
        executable='depth_converter_node',
        name='depth_converter_node',
        parameters=[config_file],
        output='screen'
    )
    
    return LaunchDescription([
        input_topic_arg,
        output_topic_arg,
        depth_converter_node,
    ])
