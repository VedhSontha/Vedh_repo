#!/usr/bin/env python3
"""
Simple Position Monitor
Displays current robot position in real-time
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
import sys

class PositionMonitor(Node):
    def __init__(self):
        super().__init__('position_monitor')
        
        # QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribe to RTABMap odometry
        self.subscription = self.create_subscription(
            Odometry,
            '/rtabmap/odom',
            self.odom_callback,
            qos_profile
        )
        
        self.get_logger().info('✅ Position Monitor started - watching /rtabmap/odom')
        self.count = 0
    
    def quaternion_to_yaw(self, x, y, z, w):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def odom_callback(self, msg):
        """Display current position"""
        self.count += 1
        
        # Extract data
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        yaw = self.quaternion_to_yaw(qx, qy, qz, qw)
        yaw_deg = math.degrees(yaw)
        
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        speed = math.sqrt(vx*vx + vy*vy)
        
        # Display every 5 messages (reduce spam)
        if self.count % 5 == 0:
            # Clear line and print (overwrites previous line)
            sys.stdout.write('\r')
            sys.stdout.write(
                f'📍 Position: x={x:7.3f}m  y={y:7.3f}m  z={z:7.3f}m  '
                f'| 🧭 Yaw: {yaw_deg:6.1f}°  '
                f'| 🚀 Speed: {speed:5.3f}m/s  '
                f'| Updates: {self.count}'
            )
            sys.stdout.flush()

def main(args=None):
    rclpy.init(args=args)
    node = PositionMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n🛑 Position Monitor stopped')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
