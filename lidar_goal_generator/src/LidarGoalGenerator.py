#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import math
from utils import gofai
from tangent_bug import bug

#RL libraries
from gym import spaces
from stable_baselines import A2C

class LidarGoalGenerator:
    def __init__(self):
        rospy.init_node('lidar_goal_generator', anonymous=True)
        
        # LiDAR subscriber
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        
        # Robot pose subscriber
        self.pose_sub = rospy.Subscriber('/odom', Odometry, self.pose_callback)

        # Robot goal subscriber
        self.goal_sub = rospy.Subscriber('/goal', Odometry, self.goal_callback)
        
        # Velocity publisher
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Parameters
        self.goal_reached_distance = 0.2  # Distance threshold to consider the goal reached
        self.linear_speed = 0.2  # Linear speed for moving towards the goal
        self.angular_speed
        
        # Goal coordinates
        self.goal_x = 0.0
        self.goal_y = 0.0
        
        # Current robot pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        
        # Flag to indicate if a goal is being navigated to
        self.navigating_to_goal = False

        #RL agent setup
        self.state = spaces.Box(low=-1, high=1, shape=((1, 4 + 2 + 12))) #todo: check if it works with only a list
        self.model = A2C.load(filepath)
        self.DWA = gofai()
        self.bug = tangent_bug()
        
    def lidar_callback(self, scan_data):
        # Process LiDAR data for RL agent
        
        min_distance = min(scan_data.ranges)
        min_distance_idx = scan_data.ranges.index(min_distance)
        angle = scan_data.angle_min + min_distance_idx * scan_data.angle_increment
        blabla=[0]*12
        
        # Set flag to navigate to the goal
        self.navigating_to_goal = True

        #write processed data to state
        self.state[6:18] = blabla
        
    def pose_callback(self, odom_data):
        # Get current robot pose
        self.robot_x = odom_data.pose.pose.position.x
        self.robot_y = odom_data.pose.pose.position.y
        self.robot_vx = odom_data.twist.twist.linear.x
        self.robot_vy = odom_data.twist.twist.linear.y

        #write robot pose and velocity to state
        self.state[2:4] = [self.robot_vx, self.robot_vy]
        self.state[4:6] = [self.robot_x, self.robot_y]
          
        # Check if robot has reached the goal
        if self.navigating_to_goal:
            distance_to_goal = math.sqrt((self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2)
            if distance_to_goal <= self.goal_reached_distance:
                self.navigating_to_goal = False
                self.publish_velocity(0.0, 0.0)  # Stop the robot
          
    def goal_callback(self, odom_data):
        # Get current robot pose
        self.goal_x = odom_data.pose.pose.position.x
        self.goal_y = odom_data.pose.pose.position.y

        #write robot pose to state
        self.state[0:2] = [self.goal_x, self.goal_y]
        
    
    def publish_velocity(self, linear, angular):
        vel_msg = Twist()
        vel_msg.linear.x = linear
        vel_msg.angular.z = angular
        self.vel_pub.publish(vel_msg)

    def action2velocity(self, action):
        """
        discretization of the position actions: they are placed in a circle around the UAV. this circle is divided by 16 for the directions.
        there are 4 circles with different radius depending on how far the drone wants to go.
        """
        duration = 0.15

        if action < settings.action_discretization * 1:
            #short distance
            speed = settings.base_speed
        elif action < settings.action_discretization * 2:
            #medium short dist
            speed = settings.base_speed*3
        elif action < settings.action_discretization * 3:
            #medium long
            speed = settings.base_speed*9
        elif action < settings.action_discretization * 4:
            #long dist
            speed = settings.base_speed*20
        else:
            print("Wrong action index!")

        action = action % settings.action_discretization
        angle = 2*math.pi/settings.action_discretization*action
        vx =  speed*math.cos(angle)
        vy = speed*math.sin(angle) #vy should be close to 0, if not, rotate:

        if vy > 0.02:
            angular = -self.angular_speed*angle #check the sign on this
            linear =  0
        else:
            linear = self.linear_speed*speed
            angular = 0


        return [linear, angular]
        
    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            if self.navigating_to_goal:
                angle_to_goal = math.atan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x)
                angular = angle_to_goal - math.atan2(math.sin(angle_to_goal - self.robot_yaw), math.cos(angle_to_goal - self.robot_yaw))

                action, _states = self.model.predict(self.state)
                local_goal = self.bug.predict(self.state)
                action = self.DWA.predict(self.state, local_goal)

                #convert to linear and angular commands
                [linear, angular] = action2velocity(action)

                self.publish_velocity(linear, angular)  # P-controller for angular velocity
            else:
                self.publish_velocity(0.0, 0.0)
            
            rate.sleep()

if __name__ == '__main__':
    try:
        lidar_goal_generator = LidarGoalGenerator()
        lidar_goal_generator.run()
    except rospy.ROSInterruptException:
        pass
