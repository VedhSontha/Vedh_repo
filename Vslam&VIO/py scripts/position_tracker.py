#!/usr/bin/env python3
"""
Robot Position Tracker
Extracts and stores the instantaneous position of the TurtleBot3
Subscribes to /odom (robot's built-in odometry) for guaranteed data
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import json
import csv
from datetime import datetime
import math

class PositionTracker(Node):
    def __init__(self):
        super().__init__('position_tracker')
        
        # Declare parameter for odometry topic
        self.declare_parameter('odom_topic', '/odom')
        odom_topic = self.get_parameter('odom_topic').value
        
        # QoS profile - try both RELIABLE and BEST_EFFORT
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Try subscribing with BEST_EFFORT first (Gazebo default)
        try:
            self.odom_subscription = self.create_subscription(
                Odometry,
                odom_topic,
                self.odom_callback,
                qos_best_effort
            )
            self.get_logger().info(f'✅ Subscribed to {odom_topic} with BEST_EFFORT QoS')
        except:
            self.odom_subscription = self.create_subscription(
                Odometry,
                odom_topic,
                self.odom_callback,
                qos_reliable
            )
            self.get_logger().info(f'✅ Subscribed to {odom_topic} with RELIABLE QoS')
        
        # Publisher for current pose (for RViz)
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            '/robot/current_pose',
            10
        )
        
        # Storage
        self.position_history = []
        self.current_position = None
        self.start_time = self.get_clock().now()
        self.last_log_time = self.start_time
        self.received_first_msg = False
        
        # Create CSV file
        self.csv_filename = f'robot_trajectory_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self.init_csv_file()
        
        # Timer for periodic logging
        self.timer = self.create_timer(0.1, self.log_position)  # 10 Hz logging
        
        # Watchdog timer to check if receiving data
        self.watchdog_timer = self.create_timer(2.0, self.check_data_reception)
        
        self.get_logger().info(f'✅ Position Tracker started')
        self.get_logger().info(f'📝 Logging to: {self.csv_filename}')
        self.get_logger().info(f'📡 Waiting for odometry data on {odom_topic}...')
        
    def check_data_reception(self):
        """Check if we're receiving data"""
        if not self.received_first_msg:
            self.get_logger().warn('⚠️  No odometry data received yet! Check if topic is publishing.')
            self.get_logger().info('   Try: ros2 topic hz /odom')
            self.get_logger().info('   Try: ros2 topic list | grep odom')
    
    def init_csv_file(self):
        """Initialize CSV file with headers"""
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 
                'elapsed_time_sec',
                'x', 'y', 'z',
                'qx', 'qy', 'qz', 'qw',
                'roll', 'pitch', 'yaw',
                'linear_vel_x', 'linear_vel_y', 'linear_vel_z',
                'angular_vel_x', 'angular_vel_y', 'angular_vel_z'
            ])
    
    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def odom_callback(self, msg):
        """Callback for odometry messages"""
        if not self.received_first_msg:
            self.received_first_msg = True
            self.get_logger().info('✅ First odometry message received!')
        
        # Extract position
        position = {
            'timestamp': self.get_clock().now().to_msg(),
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z,
            'qx': msg.pose.pose.orientation.x,
            'qy': msg.pose.pose.orientation.y,
            'qz': msg.pose.pose.orientation.z,
            'qw': msg.pose.pose.orientation.w,
            'linear_vel_x': msg.twist.twist.linear.x,
            'linear_vel_y': msg.twist.twist.linear.y,
            'linear_vel_z': msg.twist.twist.linear.z,
            'angular_vel_x': msg.twist.twist.angular.x,
            'angular_vel_y': msg.twist.twist.angular.y,
            'angular_vel_z': msg.twist.twist.angular.z,
        }
        
        # Convert quaternion to Euler angles
        roll, pitch, yaw = self.quaternion_to_euler(
            position['qx'], position['qy'], position['qz'], position['qw']
        )
        position['roll'] = roll
        position['pitch'] = pitch
        position['yaw'] = yaw
        
        self.current_position = position
        
        # Publish current pose for RViz
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose = msg.pose.pose
        self.pose_publisher.publish(pose_msg)
    
    def log_position(self):
        """Periodically log position to CSV and memory"""
        if self.current_position is None:
            return
        
        # Calculate elapsed time
        current_time = self.get_clock().now()
        elapsed = (current_time - self.start_time).nanoseconds / 1e9
        
        # Store in memory
        data_point = {
            'elapsed_time': elapsed,
            **self.current_position
        }
        self.position_history.append(data_point)
        
        # Write to CSV
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                datetime.now().isoformat(),
                f'{elapsed:.3f}',
                f'{self.current_position["x"]:.6f}',
                f'{self.current_position["y"]:.6f}',
                f'{self.current_position["z"]:.6f}',
                f'{self.current_position["qx"]:.6f}',
                f'{self.current_position["qy"]:.6f}',
                f'{self.current_position["qz"]:.6f}',
                f'{self.current_position["qw"]:.6f}',
                f'{self.current_position["roll"]:.6f}',
                f'{self.current_position["pitch"]:.6f}',
                f'{self.current_position["yaw"]:.6f}',
                f'{self.current_position["linear_vel_x"]:.6f}',
                f'{self.current_position["linear_vel_y"]:.6f}',
                f'{self.current_position["linear_vel_z"]:.6f}',
                f'{self.current_position["angular_vel_x"]:.6f}',
                f'{self.current_position["angular_vel_y"]:.6f}',
                f'{self.current_position["angular_vel_z"]:.6f}',
            ])
        
        # Print current position every 2 seconds
        time_since_last_log = (current_time - self.last_log_time).nanoseconds / 1e9
        if time_since_last_log >= 2.0:
            speed = math.sqrt(
                self.current_position["linear_vel_x"]**2 + 
                self.current_position["linear_vel_y"]**2
            )
            self.get_logger().info(
                f'📍 Position: x={self.current_position["x"]:.3f}m, '
                f'y={self.current_position["y"]:.3f}m, '
                f'yaw={math.degrees(self.current_position["yaw"]):.1f}° | '
                f'Speed: {speed:.3f}m/s | '
                f'Points: {len(self.position_history)}'
            )
            self.last_log_time = current_time
    
    def save_json(self):
        """Save trajectory to JSON file on shutdown"""
        json_filename = f'robot_trajectory_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(json_filename, 'w') as jsonfile:
            json.dump(self.position_history, jsonfile, indent=2, default=str)
        self.get_logger().info(f'💾 Saved trajectory to {json_filename}')
    
    def print_statistics(self):
        """Print trajectory statistics"""
        if len(self.position_history) < 2:
            self.get_logger().warn('Not enough data points to calculate statistics')
            return
        
        # Calculate total distance traveled
        total_distance = 0.0
        for i in range(1, len(self.position_history)):
            prev = self.position_history[i-1]
            curr = self.position_history[i]
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            dz = curr['z'] - prev['z']
            total_distance += math.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Get final position
        final = self.position_history[-1]
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('📊 Trajectory Statistics:')
        self.get_logger().info(f'   Total points logged: {len(self.position_history)}')
        self.get_logger().info(f'   Total distance: {total_distance:.3f}m')
        self.get_logger().info(f'   Duration: {final["elapsed_time"]:.2f}s')
        self.get_logger().info(f'   Final position: x={final["x"]:.3f}m, y={final["y"]:.3f}m')
        self.get_logger().info(f'   Final yaw: {math.degrees(final["yaw"]):.1f}°')
        self.get_logger().info('=' * 60)

def main(args=None):
    rclpy.init(args=args)
    node = PositionTracker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Shutting down Position Tracker...')
        node.print_statistics()
        node.save_json()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
