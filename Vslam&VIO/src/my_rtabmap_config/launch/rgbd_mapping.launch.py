from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rtabmap_ros',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[{
                'frame_id': 'base_footprint',
                'subscribe_depth': True,
                'subscribe_rgb': True,
                'approx_sync': True,
            }],
            remappings=[
                ('rgb/image', '/camera/image_raw'),
                ('depth/image', '/camera/image_raw'),
                ('camera_info', '/camera/camera_info'),
            ]
        ),
        Node(
            package='rtabmap_ros',
            executable='rtabmapviz',
            name='rtabmap_viz',
            output='screen',
        )
    ])
