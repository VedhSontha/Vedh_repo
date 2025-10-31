from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),

        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use simulation time'
        ),

        # Visual Odometry (RGB only, IMU)
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='vio_odometry',
            namespace='rtabmap',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'subscribe_rgbd': False,
                'subscribe_rgb': True,
                'subscribe_imu': True,
                'vo': True,
                'frame_id': 'base_link',
                'approx_sync': True,
            }],
            remappings=[
                ('rgb/image', '/camera/camera/image_raw'),
                ('rgb/camera_info', '/camera/camera/camera_info'),
                ('imu', '/imu'),
            ]
        ),

        # RTAB-Map SLAM (sparse)
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            namespace='rtabmap',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'subscribe_rgbd': False,
                'subscribe_rgb': True,
                'subscribe_imu': True,
                'vo': True,
                'frame_id': 'base_link',
                'approx_sync': True,
            }],
            remappings=[
                ('rgb/image', '/camera/camera/image_raw'),
                ('rgb/camera_info', '/camera/camera/camera_info'),
                ('imu', '/imu'),
            ],
            arguments=['--delete_db_on_start']
        ),
    ])

