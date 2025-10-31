#!/usr/bin/env python3
"""
Trajectory Plotter
Reads CSV file and plots the robot's trajectory
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob

def plot_trajectory(csv_file):
    """Plot robot trajectory from CSV file"""
    
    # Read CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 2D Trajectory (X-Y plane)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(df['x'], df['y'], 'b-', linewidth=2, label='Path')
    ax1.plot(df['x'].iloc[0], df['y'].iloc[0], 'go', markersize=10, label='Start')
    ax1.plot(df['x'].iloc[-1], df['y'].iloc[-1], 'ro', markersize=10, label='End')
    
    # Add arrows to show direction
    n_arrows = 10
    indices = np.linspace(0, len(df)-1, n_arrows, dtype=int)
    for i in indices[:-1]:
        dx = df['x'].iloc[i+1] - df['x'].iloc[i]
        dy = df['y'].iloc[i+1] - df['y'].iloc[i]
        ax1.arrow(df['x'].iloc[i], df['y'].iloc[i], dx, dy, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5)
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('2D Trajectory (Top View)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Position vs Time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(df['elapsed_time_sec'], df['x'], 'r-', label='X', linewidth=2)
    ax2.plot(df['elapsed_time_sec'], df['y'], 'g-', label='Y', linewidth=2)
    ax2.plot(df['elapsed_time_sec'], df['z'], 'b-', label='Z', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Yaw Angle vs Time
    ax3 = plt.subplot(2, 3, 3)
    yaw_deg = np.degrees(df['yaw'])
    ax3.plot(df['elapsed_time_sec'], yaw_deg, 'purple', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Yaw Angle (degrees)')
    ax3.set_title('Orientation vs Time')
    ax3.grid(True, alpha=0.3)
    
    # 4. Linear Velocity
    ax4 = plt.subplot(2, 3, 4)
    speed = np.sqrt(df['linear_vel_x']**2 + df['linear_vel_y']**2)
    ax4.plot(df['elapsed_time_sec'], speed, 'orange', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Linear Speed vs Time')
    ax4.grid(True, alpha=0.3)
    
    # 5. Angular Velocity
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(df['elapsed_time_sec'], df['angular_vel_z'], 'brown', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angular Velocity (rad/s)')
    ax5.set_title('Angular Velocity (Yaw) vs Time')
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistics Text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate statistics
    total_distance = 0
    for i in range(1, len(df)):
        dx = df['x'].iloc[i] - df['x'].iloc[i-1]
        dy = df['y'].iloc[i] - df['y'].iloc[i-1]
        dz = df['z'].iloc[i] - df['z'].iloc[i-1]
        total_distance += np.sqrt(dx**2 + dy**2 + dz**2)
    
    duration = df['elapsed_time_sec'].iloc[-1]
    avg_speed = np.mean(speed)
    max_speed = np.max(speed)
    
    stats_text = f"""
    TRAJECTORY STATISTICS
    ═══════════════════════
    
    Total Points: {len(df)}
    Duration: {duration:.2f} seconds
    
    Total Distance: {total_distance:.3f} m
    Average Speed: {avg_speed:.3f} m/s
    Max Speed: {max_speed:.3f} m/s
    
    Start Position:
      x = {df['x'].iloc[0]:.3f} m
      y = {df['y'].iloc[0]:.3f} m
    
    End Position:
      x = {df['x'].iloc[-1]:.3f} m
      y = {df['y'].iloc[-1]:.3f} m
    
    Displacement:
      Δx = {df['x'].iloc[-1] - df['x'].iloc[0]:.3f} m
      Δy = {df['y'].iloc[-1] - df['y'].iloc[0]:.3f} m
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(csv_file.replace('.csv', '_plot.png'), dpi=150)
    print(f"✅ Plot saved as: {csv_file.replace('.csv', '_plot.png')}")
    plt.show()

def main():
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Find the most recent CSV file
        csv_files = glob.glob('robot_trajectory_*.csv')
        if not csv_files:
            print("❌ No trajectory CSV files found!")
            print("Usage: python3 plot_trajectory.py <csv_file>")
            return
        csv_file = max(csv_files, key=lambda x: x)
        print(f"📂 Using most recent file: {csv_file}")
    
    plot_trajectory(csv_file)

if __name__ == '__main__':
    main()
