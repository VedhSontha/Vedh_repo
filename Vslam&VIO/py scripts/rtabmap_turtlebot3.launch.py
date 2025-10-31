from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        
        # RGB-D Odometry Node
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='rgbd_odometry',
            namespace='rtabmap',
            parameters=[{
                'use_sim_time': use_sim_time,
                'sync_queue_size': 20,
                'approx_sync': True
            }],
            remappings=[
                ('rgb/image', '/camera/camera/image_raw'),
                ('depth/image', '/depth/camera/depth/image_raw'),
                ('rgb/camera_info', '/depth/camera/depth/camera_info'),
            ],
            output='screen'
        ),
        
        # RTAB-Map SLAM Node
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            namespace='rtabmap',
            parameters=[{
                'use_sim_time': use_sim_time,
                'subscribe_depth': True,
                'subscribe_rgb': True,
                'subscribe_scan': False,
                'frame_id': 'base_footprint',
                'approx_sync': True,
                'sync_queue_size': 50,
            }],
            remappings=[
                ('rgb/image', '/camera/camera/image_raw'),
                ('depth/image', '/depth/camera/depth/image_raw'),
                ('rgb/camera_info', '/depth/camera/depth/camera_info'),
            ],
            output='screen',
            arguments=['--delete_db_on_start']
        ),
        
        # RTAB-Map Visualization
        Node(
            package='rtabmap_viz',
            executable='rtabmap_viz',
            name='rtabmap_viz',
            namespace='rtabmap',
            parameters=[{
                'use_sim_time': use_sim_time,
                'sync_queue_size': 20,
            }],
            remappings=[
                ('rgb/image', '/camera/camera/image_raw'),
                ('depth/image', '/depth/camera/depth/image_raw'),
                ('rgb/camera_info', '/depth/camera/depth/camera_info'),
            ],
            output='screen'
        ),
    ])
