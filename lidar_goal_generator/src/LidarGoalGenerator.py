#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped

class LidarGoalGenerator:
    def __init__(self):
        rospy.init_node('lidar_goal_generator', anonymous=True)
        
        # LiDAR subscriber
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        
        # Goal publisher
        self.goal_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)
        
    def lidar_callback(self, scan_data):
        # Process LiDAR data and generate goal coordinates
        # For example, find the minimum distance and angle of the nearest obstacle
        
        min_distance = min(scan_data.ranges)
        min_distance_idx = scan_data.ranges.index(min_distance)
        angle = scan_data.angle_min + min_distance_idx * scan_data.angle_increment
        
        # Generate goal coordinates based on the minimum distance and angle
        goal_pose = PoseStamped()
        goal_pose.header.stamp = rospy.Time.now()
        goal_pose.header.frame_id = 'base_link'  # Assuming the base frame
        
        goal_pose.pose.position.x = min_distance * cos(angle)
        goal_pose.pose.position.y = min_distance * sin(angle)
        goal_pose.pose.orientation.w = 1.0  # No rotation
        
        # Publish goal coordinates
        self.goal_pub.publish(goal_pose)
        
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        lidar_goal_generator = LidarGoalGenerator()
        lidar_goal_generator.run()
    except rospy.ROSInterruptException:
        pass

