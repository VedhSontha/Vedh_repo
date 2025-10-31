from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    parameters = [{
        'frame_id': 'base_footprint',
        'subscribe_depth': False,
        'subscribe_rgb': True,
        'subscribe_scan': False,
        'use_action_for_goal': True,
        'qos_image': 1,
        'qos_imu': 1,
        'Reg/Force3DoF': 'false',  # Allow 3D (changed from true)
        'Optimizer/GravitySigma': '0.3',
        
        # Monocular settings
        'Vis/EstimationType': '0',  # 0=3D->3D
        'Vis/MinInliers': '15',
        'Vis/InlierDistance': '0.1',
        'Vis/FeatureType': '6',  # 0=SURF, 6=GFTT, 8=ORB
        'Vis/MaxFeatures': '500',
        
        # 3D Point cloud generation from monocular
        'Stereo/MaxDisparity': '200.0',
        'Stereo/OpticalFlow': 'true',  # Use optical flow for depth
        'Vis/DepthAsMask': 'false',
        
        # Memory management
        'Rtabmap/DetectionRate': '1.0',
        'Mem/RehearsalSimilarity': '0.45',
        'Mem/ImageKept': 'false',
        'Mem/STMSize': '30',
        'Mem/IncrementalMemory': 'true',
        
        # Visual odometry
        'Odom/Strategy': '0',  # 0=Frame-to-Map
        'OdomF2M/MaxSize': '1000',
        'Odom/GuessMotion': 'true',
        'Odom/ResetCountdown': '15',
        
        # 3D Cloud output
        'Grid/FromDepth': 'false',
        'Grid/3D': 'true',
        'Grid/RangeMax': '5.0',
        'Grid/CellSize': '0.05',
    }]

    remappings = [
        ('rgb/image', '/camera/image_raw'),
        ('rgb/camera_info', '/camera/camera_info'),
        ('imu', '/imu'),
    ]

    return LaunchDescription([
        # RTAB-Map SLAM node
        Node(
            package='rtabmap_slam', executable='rtabmap',
            output='screen',
            parameters=parameters,
            remappings=remappings,
            arguments=['-d']),  # -d to delete old database
            
        # Visual odometry node
        Node(
            package='rtabmap_odom', executable='rgbd_odometry',
            output='screen',
            parameters=parameters,
            remappings=remappings),
            
        # Point cloud assembler
        Node(
            package='rtabmap_util', executable='point_cloud_assembler',
            output='screen',
            parameters=[{
                'max_clouds': 10,
                'fixed_frame_id': 'odom'
            }],
            remappings=[
                ('cloud', 'voxel_cloud')
            ]),
            
        # Visualization
        Node(
            package='rtabmap_viz', executable='rtabmap_viz',
            output='screen',
            parameters=parameters,
            remappings=remappings),
    ])
