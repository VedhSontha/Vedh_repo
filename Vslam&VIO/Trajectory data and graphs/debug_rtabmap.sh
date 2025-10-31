#!/bin/bash

echo "================================"
echo "RTABMap Debugging Script"
echo "================================"
echo ""

echo "1. Checking ROS2 nodes..."
ros2 node list
echo ""

echo "2. Checking camera topics..."
ros2 topic list | grep camera
echo ""

echo "3. Checking RTABMap topics..."
ros2 topic list | grep rtabmap
echo ""

echo "4. Checking camera data rate..."
timeout 3 ros2 topic hz /camera/image_raw 2>&1 || echo "⚠️  No data on /camera/image_raw"
echo ""

echo "5. Checking camera_info..."
timeout 2 ros2 topic echo /camera/camera_info --no-arr 2>&1 | head -5 || echo "⚠️  No camera_info"
echo ""

echo "6. Checking fake IMU..."
timeout 2 ros2 topic echo /imu_throttled --no-arr 2>&1 | head -5 || echo "⚠️  No IMU data"
echo ""

echo "7. Checking fake depth..."
timeout 2 ros2 topic echo /rtabmap/depth/image --no-arr 2>&1 | head -5 || echo "⚠️  No depth data"
echo ""

echo "8. Checking odometry..."
timeout 2 ros2 topic echo /rtabmap/odom --no-arr 2>&1 | head -5 || echo "⚠️  No odometry"
echo ""

echo "9. Checking TF tree..."
ros2 run tf2_tools view_frames
echo "TF tree saved to frames.pdf"
echo ""

echo "================================"
echo "Debug complete!"
echo "================================"
