#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class ImuThrottle(Node):
    def __init__(self):
        super().__init__('imu_throttle')
        
        # Declare parameters
        self.declare_parameter('target_rate', 50.0)
        self.declare_parameter('input_topic', '/imu')
        self.declare_parameter('output_topic', '/imu_throttled')
        
        # Get parameters
        target_rate = self.get_parameter('target_rate').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        
        self.min_interval = 1.0 / target_rate
        self.last_pub_time = None
        
        # QoS settings
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create subscriber and publisher
        self.sub = self.create_subscription(
            Imu,
            input_topic,
            self.imu_callback,
            qos
        )
        
        self.pub = self.create_publisher(Imu, output_topic, qos)
        
        self.get_logger().info(f'Throttling {input_topic} to {target_rate} Hz -> {output_topic}')
    
    def imu_callback(self, msg):
        current_time = self.get_clock().now()
        
        if self.last_pub_time is None:
            self.pub.publish(msg)
            self.last_pub_time = current_time
            return
        
        time_diff = (current_time - self.last_pub_time).nanoseconds / 1e9
        
        if time_diff >= self.min_interval:
            self.pub.publish(msg)
            self.last_pub_time = current_time

def main(args=None):
    rclpy.init(args=args)
    node = ImuThrottle()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
