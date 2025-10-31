#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class FakeDepthPublisher(Node):
    def __init__(self):
        super().__init__('fake_depth_publisher')
        self.bridge = CvBridge()
        
        # Subscribe to the RGB camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        
        # Publisher for fake depth image
        self.publisher = self.create_publisher(
            Image,
            '/rtabmap/depth/image',
            10)
        
        self.get_logger().info('✅ FakeDepthPublisher started and waiting for /camera/image_raw...')
    
    def image_callback(self, msg):
        """Generate a zero-filled depth image that matches the RGB frame."""
        # Create zero depth image
        fake_depth = np.zeros((msg.height, msg.width), dtype=np.float32)
        
        # Convert to ROS message
        depth_msg = self.bridge.cv2_to_imgmsg(fake_depth, encoding='32FC1')
        
        # CRITICAL FIX: Copy exact timestamp and frame ID for proper sync
        depth_msg.header.stamp = msg.header.stamp
        depth_msg.header.frame_id = msg.header.frame_id
        
        # Publish immediately to minimize latency
        self.publisher.publish(depth_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FakeDepthPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # FIX: Only shutdown if rclpy is still running
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
