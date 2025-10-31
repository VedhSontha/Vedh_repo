from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    parameters = [{
        'frame_id': 'base_footprint',
        'subscribe_depth': False,
        'subscribe_rgb': True,
        'subscribe_scan': False,
        'wait_imu_to_init': True,
        
        # VIO settings
        'Odom/Strategy': '0',
        'Odom/GuessMotion': 'true',
        'Vis/EstimationType': '0',
        'Vis/FeatureType': '6',  # GFTT
        'Vis/MaxFeatures': '1000',
        'Vis/MinInliers': '20',
        
        'RGBD/CreateOccupancyGrid': 'false',
        'Rtabmap/CreateIntermediateNodes': 'true',
    }]

    remappings = [
        ('rgb/image', '/camera/image_raw'),
        ('rgb/camera_info', '/camera/camera_info'),
        ('imu', '/imu'),
    ]

    return LaunchDescription([
        # Visual-inertial odometry
        Node(
            package='rtabmap_odom', executable='rgbd_odometry',
            output='screen',
            parameters=parameters,
            remappings=remappings),
            
        # RTAB-Map for loop closure
        Node(
            package='rtabmap_slam', executable='rtabmap',
            output='screen',
            parameters=parameters,
            remappings=remappings,
            arguments=['-d']),
            
        # Visualization
        Node(
            package='rtabmap_viz', executable='rtabmap_viz',
            output='screen',
            parameters=parameters,
            remappings=remappings),
    ])
