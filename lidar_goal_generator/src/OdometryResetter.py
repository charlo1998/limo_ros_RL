#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler

class OdometryResetter:
    def __init__(self):
        rospy.init_node('odometry_resetter')
        
        # Service server to reset odometry
        self.reset_service = rospy.Service('/reset_odometry', Empty, self.reset_odometry)
        
        # Odometry publisher
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        
        # Initial robot pose
        self.initial_x = 0.0
        self.initial_y = 0.0
        self.initial_yaw = 0.0
        
    def reset_odometry(self, req):
        # Reset the robot's odometry to the initial pose
        
        # Create a new Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        
        # Set the pose
        odom_msg.pose.pose.position.x = self.initial_x
        odom_msg.pose.pose.position.y = self.initial_y
        odom_msg.pose.pose.position.z = 0.0
        orientation = quaternion_from_euler(0.0, 0.0, self.initial_yaw)
        odom_msg.pose.pose.orientation.x = orientation[0]
        odom_msg.pose.pose.orientation.y = orientation[1]
        odom_msg.pose.pose.orientation.z = orientation[2]
        odom_msg.pose.pose.orientation.w = orientation[3]
        
        # Publish the reset odometry message
        self.odom_pub.publish(odom_msg)
        
        return []
    
    def run(self):
        rospy.spin()
        print("running")

if __name__ == '__main__':
    try:
        odometry_resetter = OdometryResetter()
        odometry_resetter.run()
    except rospy.ROSInterruptException:
        pass
