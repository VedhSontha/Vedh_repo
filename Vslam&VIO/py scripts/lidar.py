"""
Standalone RTABMap Launch for TurtleBot3 with LiDAR
No package required - run directly with ros2 launch
Assumes fake_depth_publisher.py is already running
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        
        # ========================================
        # VISUAL ODOMETRY NODE
        # ========================================
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='visual_odometry',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'frame_id': 'camera_rgb_optical_frame',
                'odom_frame_id': 'odom',
                'publish_tf': True,
                'wait_for_transform': 1.0,
                
                # QoS for Gazebo (BEST_EFFORT)
                'qos': 1,
                'qos_image': 1,
                'qos_camera_info': 1,
                
                # Synchronization
                'approx_sync': True,
                'queue_size': 30,
                'approx_sync_max_interval': 0.2,  # Increased tolerance
                
                # Subscription
                'subscribe_rgbd': False,
                'subscribe_rgb': True,
                'subscribe_depth': True,
                'subscribe_scan': False,
                'wait_imu_to_init': False,  # NO IMU required
                
                # Odometry
                'Odom/Strategy': '0',
                'Odom/ResetCountdown': '1',
                'Odom/FillInfoData': 'true',
                'Odom/GuessMotion': 'true',
                
                # Visual features
                'Vis/EstimationType': '1',
                'Vis/FeatureType': '6',  # GFTT/BRIEF
                'Vis/MaxFeatures': '400',
                'Vis/MinInliers': '15',
                'Vis/InlierDistance': '0.1',
                'Vis/MaxDepth': '0',  # Ignore depth values
            }],
            remappings=[
                ('rgb/image', '/camera/image_raw'),
                ('rgb/camera_info', '/camera/camera_info'),
                ('depth/image', '/rtabmap/depth/image'),
                ('odom', '/rtabmap/odom'),
            ]
        ),
        
        # ========================================
        # RTAB-MAP SLAM WITH LIDAR
        # ========================================
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                
                # Subscription
                'subscribe_depth': False,
                'subscribe_rgb': True,
                'subscribe_stereo': False,
                'subscribe_rgbd': False,
                'subscribe_scan': True,  # LIDAR ENABLED
                'subscribe_scan_cloud': False,
                'subscribe_odom_info': True,
                
                # Frames
                'frame_id': 'camera_rgb_optical_frame',
                'map_frame_id': 'map',
                'odom_frame_id': 'odom',
                'publish_tf': True,
                
                # QoS for Gazebo
                'qos': 1,
                'qos_image': 1,
                'qos_camera_info': 1,
                'qos_scan': 1,
                
                # Synchronization
                'queue_size': 30,
                'approx_sync': True,
                
                # RGBD disabled (visual SLAM)
                'RGBD/Enabled': 'false',
                'RGBD/LinearUpdate': '0.1',
                'RGBD/AngularUpdate': '0.1',
                
                # Memory
                'Mem/IncrementalMemory': 'true',
                'Mem/InitWMWithAllNodes': 'false',
                'Mem/STMSize': '30',
                
                # Loop closure
                'Rtabmap/DetectionRate': '1.0',
                'Rtabmap/TimeThr': '0',
                
                # Visual
                'Vis/MinInliers': '15',
                'Vis/InlierDistance': '0.1',
                
                # ===== LIDAR FOR POINT CLOUD =====
                'Grid/FromDepth': 'false',  # Don't use fake depth
                'Grid/Sensor': '1',  # 1 = Use laser scan
                'Grid/3D': 'true',  # Enable 3D mapping
                
                # Laser scan parameters
                'Grid/RangeMin': '0.2',  # Minimum range (m)
                'Grid/RangeMax': '12.0',  # Maximum range for TurtleBot3
                'Grid/CellSize': '0.05',  # 5cm grid cells
                
                # Obstacle detection
                'Grid/MaxObstacleHeight': '2.0',  # Max height (m)
                'Grid/MaxGroundHeight': '0.0',  # Ground level
                'Grid/GroundIsObstacle': 'false',
                'Grid/ClusterRadius': '0.1',
                'Grid/MinClusterSize': '10',
                'Grid/FlatObstacleDetected': 'true',
                
                # Point cloud output
                'cloud_output_voxelized': 'false',
            }],
            remappings=[
                ('rgb/image', '/camera/image_raw'),
                ('rgb/camera_info', '/camera/camera_info'),
                ('odom', '/rtabmap/odom'),
                ('scan', '/scan'),  # TurtleBot3 LiDAR
            ],
            arguments=['--delete_db_on_start']
        ),
        
        # ========================================
        # RTABMAP VISUALIZATION
        # ========================================
        Node(
            package='rtabmap_viz',
            executable='rtabmap_viz',
            name='rtabmap_viz',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'frame_id': 'camera_rgb_optical_frame',
                'odom_frame_id': 'odom',
                'subscribe_odom_info': True,
                'subscribe_rgb': True,
                'subscribe_depth': False,
                'subscribe_scan': True,  # Show LiDAR
                'approx_sync': True,
                'queue_size': 30,
                'qos': 1,
                'qos_image': 1,
                'qos_camera_info': 1,
                'qos_scan': 1,
            }],
            remappings=[
                ('rgb/image', '/camera/image_raw'),
                ('rgb/camera_info', '/camera/camera_info'),
                ('odom', '/rtabmap/odom'),
                ('scan', '/scan'),
            ]
        ),
    ])
