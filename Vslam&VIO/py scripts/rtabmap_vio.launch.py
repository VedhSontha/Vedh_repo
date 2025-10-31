from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # Get the paths to the fake publisher scripts
    fake_depth_script = os.path.join(os.getcwd(), 'fake_depth_publisher.py')
    fake_imu_script = os.path.join(os.getcwd(), 'fake_imu_publisher.py')
    
    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        
        # ========================================
        # FAKE PUBLISHERS
        # ========================================
        
        # Fake IMU publisher - Provides orientation for odometry initialization
        ExecuteProcess(
            cmd=['python3', fake_imu_script],
            output='screen',
            name='fake_imu_publisher'
        ),
        
        # Fake depth publisher - Creates zero depth image for visual-only odometry
        ExecuteProcess(
            cmd=['python3', fake_depth_script],
            output='screen',
            name='fake_depth_publisher'
        ),
        
        # ========================================
        # VISUAL ODOMETRY NODE
        # ========================================
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='visual_odometry',
            namespace='rtabmap',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'frame_id': 'camera_rgb_optical_frame',
                'odom_frame_id': 'odom',
                'publish_tf': True,
                'wait_for_transform': 2.0,
                
                # FIXED: Improved synchronization parameters
                'approx_sync': True,
                'sync_queue_size': 50,  # Reduced from 200 for better sync
                'topic_queue_size': 10,
                'qos': 2,
                'approx_sync_max_interval': 0.02,  # FIXED: Lowered from 0.5 to 0.02 (20ms)
                
                # Subscription settings
                'subscribe_rgbd': False,
                'subscribe_rgb': True,
                'subscribe_depth': True,
                'wait_imu_to_init': True,  # Now safe - we have fake IMU
                
                # Odometry strategy
                'Odom/Strategy': '0',  # 0=Frame-to-Map, 1=Frame-to-Frame
                'Odom/ResetCountdown': '1',
                'Odom/ImageDecimation': '1',
                'Odom/FillInfoData': 'true',
                'Odom/GuessMotion': 'true',
                
                # Visual features configuration
                'Vis/EstimationType': '1',  # 1=PnP (3D->2D)
                'Vis/FeatureType': '8',  # 8=FAST/BRIEF
                'Vis/MaxFeatures': '1000',
                'Vis/MinInliers': '15',
                'Vis/InlierDistance': '0.1',
                
                # RGBD specific (even though depth is zero, these help stability)
                'Vis/DepthAsMask': 'false',  # Don't use depth as mask
                'Vis/MaxDepth': '0',  # Ignore depth values
                
                # Keyframe selection
                'Odom/KeyFrameThr': '0.3',  # New keyframe if transform > 30%
            }],
            remappings=[
                ('rgb/image', '/camera/image_raw'),
                ('rgb/camera_info', '/camera/camera_info'),
                ('depth/image', '/rtabmap/depth/image'),
                ('imu', '/imu_throttled'),
            ]
        ),
        
        # ========================================
        # RTAB-MAP SLAM NODE
        # ========================================
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            namespace='rtabmap',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                
                # Subscription settings
                'subscribe_depth': False,
                'subscribe_rgb': True,
                'subscribe_stereo': False,
                'subscribe_rgbd': False,
                'subscribe_odom_info': True,
                
                # Frame IDs
                'frame_id': 'camera_rgb_optical_frame',
                'map_frame_id': 'map',
                'odom_frame_id': 'odom',
                'publish_tf': True,
                
                # FIXED: Improved synchronization parameters
                'sync_queue_size': 50,  # Reduced from 200
                'topic_queue_size': 10,
                'qos': 2,
                'approx_sync': True,
                'approx_sync_max_interval': 0.02,  # FIXED: Lowered from 0.5
                
                # Disable RGB-D mode (we're doing visual SLAM)
                'RGBD/Enabled': 'false',
                'RGBD/LinearUpdate': '0.1',
                'RGBD/AngularUpdate': '0.1',
                
                # Memory management
                'Mem/IncrementalMemory': 'true',
                'Mem/InitWMWithAllNodes': 'false',
                'Mem/STMSize': '30',  # Short-term memory size
                
                # Loop closure detection
                'Rtabmap/DetectionRate': '1.0',  # Hz
                'Rtabmap/TimeThr': '0',  # No time limit for loop closure
                
                # Visual parameters
                'Vis/MinInliers': '15',
                'Vis/InlierDistance': '0.1',
            }],
            remappings=[
                ('rgb/image', '/camera/image_raw'),
                ('rgb/camera_info', '/camera/camera_info'),
                ('odom', '/rtabmap/odom'),
            ],
            arguments=['--delete_db_on_start']
        ),
        
        # ========================================
        # VISUALIZATION (Optional but helpful)
        # ========================================
        Node(
            package='rtabmap_viz',
            executable='rtabmap_viz',
            name='rtabmap_viz',
            namespace='rtabmap',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'frame_id': 'camera_rgb_optical_frame',
                'odom_frame_id': 'odom',
                'subscribe_odom_info': True,
                'subscribe_rgb': True,
                'subscribe_depth': False,
                'approx_sync': True,
            }],
            remappings=[
                ('rgb/image', '/camera/image_raw'),
                ('rgb/camera_info', '/camera/camera_info'),
                ('odom', '/rtabmap/odom'),
            ]
        ),
    ])
