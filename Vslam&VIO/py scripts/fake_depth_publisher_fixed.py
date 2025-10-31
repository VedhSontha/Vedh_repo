#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class FakeDepthPublisher(Node):
    def __init__(self):
        super().__init__('fake_depth_publisher')
        self.bridge = CvBridge()
        
        # QoS profile matching Gazebo (BEST_EFFORT)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to the RGB camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile)
        
        # Publisher for fake depth image
        self.publisher = self.create_publisher(
            Image,
            '/rtabmap/depth/image',
            qos_profile)
        
        self.get_logger().info('✅ FakeDepthPublisher started - waiting for /camera/image_raw')
        self.msg_count = 0
    
    def image_callback(self, msg):
        """Generate a zero-filled depth image that matches the RGB frame."""
        # Create zero depth image
        fake_depth = np.zeros((msg.height, msg.width), dtype=np.float32)
        
        # Convert to ROS message
        depth_msg = self.bridge.cv2_to_imgmsg(fake_depth, encoding='32FC1')
        
        # CRITICAL: Copy exact timestamp and frame ID for proper sync
        depth_msg.header.stamp = msg.header.stamp
        depth_msg.header.frame_id = msg.header.frame_id
        
        # Publish immediately
        self.publisher.publish(depth_msg)
        
        # Log occasionally
        self.msg_count += 1
        if self.msg_count % 100 == 0:
            self.get_logger().info(f'Published {self.msg_count} fake depth images')

def main(args=None):
    rclpy.init(args=args)
    node = FakeDepthPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
