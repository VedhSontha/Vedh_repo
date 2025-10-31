from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    parameters = [{
        'frame_id': 'base_footprint',
        'subscribe_depth': True,
        'subscribe_rgb': True,
        'subscribe_scan': False,
        'approx_sync': True,
        'queue_size': 10,
        
        # RGBD specific
        'Rtabmap/DetectionRate': '1.0',
        'RGBD/NeighborLinkRefining': 'true',
        'RGBD/ProximityBySpace': 'true',
        'RGBD/AngularUpdate': '0.01',
        'RGBD/LinearUpdate': '0.01',
        'RGBD/OptimizeFromGraphEnd': 'false',
        'Reg/Strategy': '1',  # 0=Vis, 1=ICP, 2=Vis+ICP
        'Reg/Force3DoF': 'false',
        'Optimizer/Slam2D': 'false',
        
        # Memory
        'Mem/RehearsalSimilarity': '0.30',
        'Mem/STMSize': '30',
        'Mem/ImageKept': 'false',
        
        # 3D Cloud generation
        'Grid/FromDepth': 'true',
        'Grid/3D': 'true',
        'Grid/RangeMax': '5.0',
        'Grid/RangeMin': '0.2',
        'Grid/CellSize': '0.05',
        'Grid/ClusterRadius': '1.0',
        'Grid/GroundIsObstacle': 'false',
        'Grid/MaxGroundHeight': '0.0',
        'Grid/MaxObstacleHeight': '2.0',
        
        # Visual features
        'Vis/MinInliers': '12',
        'Vis/MaxFeatures': '500',
    }]

    remappings = [
        ('rgb/image', '/camera/image_raw'),
        ('rgb/camera_info', '/camera/camera_info'),
        ('depth/image', '/depth/image'),
        ('imu', '/imu'),
    ]

    return LaunchDescription([
        # RTAB-Map SLAM node
        Node(
            package='rtabmap_slam', executable='rtabmap',
            output='screen',
            parameters=parameters,
            remappings=remappings,
            arguments=['-d']),  # Delete old database
            
        # RGB-D odometry
        Node(
            package='rtabmap_odom', executable='rgbd_odometry',
            output='screen',
            parameters=parameters,
            remappings=remappings),
            
        # Visualization
        Node(
            package='rtabmap_viz', executable='rtabmap_viz',
            output='screen',
            parameters=parameters,
            remappings=remappings),
    ])
