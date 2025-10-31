#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class FakeIMUPublisher(Node):
    def __init__(self):
        super().__init__('fake_imu_publisher')
        
        # Configure QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publisher for fake IMU data
        self.publisher = self.create_publisher(
            Imu,
            '/imu_throttled',
            qos_profile)
        
        # CRITICAL: 20 Hz to match camera rate (NOT 100 Hz)
        self.timer = self.create_timer(0.05, self.publish_imu)  # 20 Hz
        
        self.get_logger().info('✅ Fake IMU Publisher started on /imu_throttled at 20 Hz')
    
    def publish_imu(self):
        """Publish a fake IMU message with proper orientation."""
        msg = Imu()
        
        # Set timestamp
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_rgb_optical_frame'  # CRITICAL: Same as camera frame
        
        # Identity orientation (no rotation)
        msg.orientation.w = 1.0
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        
        # CRITICAL: Non-zero covariance diagonal (ROS convention for "set")
        msg.orientation_covariance[0] = 0.0001  # xx
        msg.orientation_covariance[4] = 0.0001  # yy
        msg.orientation_covariance[8] = 0.0001  # zz
        # All other elements remain 0
        
        # Zero angular velocity
        msg.angular_velocity.x = 0.0
        msg.angular_velocity.y = 0.0
        msg.angular_velocity.z = 0.0
        msg.angular_velocity_covariance[0] = 0.0001
        msg.angular_velocity_covariance[4] = 0.0001
        msg.angular_velocity_covariance[8] = 0.0001
        
        # Zero linear acceleration (except gravity)
        msg.linear_acceleration.x = 0.0
        msg.linear_acceleration.y = 0.0
        msg.linear_acceleration.z = 9.81  # Gravity
        msg.linear_acceleration_covariance[0] = 0.0001
        msg.linear_acceleration_covariance[4] = 0.0001
        msg.linear_acceleration_covariance[8] = 0.0001
        
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FakeIMUPublisher()
    
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
