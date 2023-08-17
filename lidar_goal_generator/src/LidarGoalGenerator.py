#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import math
from utils import gofai
from tangent_bug import tangent_bug
import numpy as np
from bisect import bisect

#RL libraries
from stable_baselines3 import A2C

class LidarGoalGenerator:
    def __init__(self, filepath):
        rospy.init_node('lidar_goal_generator', anonymous=True)
        
        # LiDAR subscriber
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        
        # Robot pose subscriber
        self.pose_sub = rospy.Subscriber('/odom', Odometry, self.pose_callback)

        # Robot goal subscriber
        #self.goal_sub = rospy.Subscriber('/goal', Odometry, self.goal_callback)
        
        # Velocity publisher
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # goal publisher
        self.goal_pub = rospy.Publisher('/goal', Odometry, queue_size=10)
        
        # Parameters
        self.goal_reached_distance = 0.2  # Distance threshold to consider the goal reached
        self.linear_speed = 0.2  # Linear speed for moving towards the goal
        self.angular_speed
        
        # Goal coordinates
        self.goal_x = 3.0
        self.goal_y = 1.0
        
        # Current robot pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        
        # Flag to indicate if a goal is being navigated to
        self.navigating_to_goal = True

        #RL agent setup
        self.nb_of_sensors = 12
        self.state = np.zeros((1, 1, 4 + 2 + 12)) #todo: check if it works with only a list
        self.model = A2C.load(filepath)
        self.DWA = gofai()
        self.bug = tangent_bug()
        
    def lidar_callback(self, scan_data):
        # Process LiDAR data for RL agent
        
        distances = scan_data.ranges
        angles = scan_data.angle_min + distances * scan_data.angle_increment
        
        left_angle=-180#keep the same convention as the agent learned, even tho the lidar won't fill the whole circle.
        right_angle=180

        lidar_FOV =  math.ceil(scan_data.angle_max - scan_data.angle_min)

        #spliting points into ranges of angles
        theta = 360/self.nb_of_sensors
        for i in range(self.nb_of_sensors):
            if i == 0:
                thetas = [-180]
            else:
                thetas.append(thetas[i-1]+theta)

        if len(angles) == 0: #lidar not seeing anything!
            #np.concatenate((np.zeros(nb_of_sensors), np.array(thetas)/180.0))
            return np.zeros(self.nb_of_sensors)

        #print(f"angle ranges: {thetas}")
        #print(f"angle left: {angle_left}")
        #print(f"angle right: {angle_right}")
        #print(f"number of points: {len(angles)}")

        #adding points
        distances_by_sensor = [[] for i in range(self.nb_of_sensors)]

        for i, angle in enumerate(angles):
            if (angle > left_angle and angle < right_angle): #place only the points inside the input FOV
                ith_sensor = bisect(thetas,angle) #the bisect fnc finds where the angle would fit in the ranges we created (thetas)
                distances_by_sensor[ith_sensor-1].append(distances[i])

        sensors = [0 for x in range(self.nb_of_sensors)]

        for i in range(self.nb_of_sensors):
            if len(distances_by_sensor[i]) == 0: #missing lidar values!
                sensors[i] = 66 #with no information, set to 66, which will be ignored
                #print(f"missing lidar values in bucket {i}!")
            else:
                sensors[i] = min(distances_by_sensor[i])
            #normalizing values and bounding them to [-1,1]
            sensors[i] = np.log(sensors[i]+0.0001)/np.log(100) #this way gives more range to the smaller distances (large distances are less important).
            sensors[i] = min(1,max(-1,sensors[i]))

        #write processed data to state
        self.state[6:18] = sensors

        
    def pose_callback(self, odom_data):
        # Get current robot pose
        self.robot_x = odom_data.pose.pose.position.x
        self.robot_y = odom_data.pose.pose.position.y
        self.robot_vx = odom_data.twist.twist.linear.x
        self.robot_vy = odom_data.twist.twist.linear.y

        #write robot pose and velocity to state
        self.state[2:4] = [self.robot_vx, self.robot_vy]
        self.state[4:6] = [self.robot_x, self.robot_y]
          
          
    # def goal_callback(self, odom_data):
    #     # Get current robot pose
    #     self.goal_x = odom_data.pose.pose.position.x
    #     self.goal_y = odom_data.pose.pose.position.y

    #     #write robot pose to state
    #     self.state[0:2] = [self.goal_x, self.goal_y]
        
    
    def publish_velocity(self, linear, angular):
        vel_msg = Twist()
        vel_msg.linear.x = linear
        vel_msg.angular.z = angular
        self.vel_pub.publish(vel_msg)

    def publish_goal(self, x, y):
        goal_msg = Odometry()
        goal_msg.pose.pose.position.x = self.goal_x
        goal_msg.pose.pose.position.y = self.goal_y

        #write robot pose to state directly
        self.state[0:2] = [self.goal_x, self.goal_y]

        self.goal_pub.publish(goal_msg)


    def action2velocity(self, action):
        """
        discretization of the position actions: they are placed in a circle around the UAV. this circle is divided by 16 for the directions.
        there are 4 circles with different radius depending on how far the drone wants to go.
        """
        duration = 0.15
        speed = self.linear_speed

        if action < self.nb_of_sensors * 1:
            #short distance
            speed *= 1
        elif action < self.nb_of_sensors * 2:
            #medium short dist
            speed *= 3
        elif action < self.nb_of_sensors * 3:
            #medium long
            speed *= 9
        elif action < self.nb_of_sensors * 4:
            #long dist
            speed *= 20
        else:
            print("Wrong action index!")

        action = action % self.nb_of_sensors
        angle = 2*math.pi/self.nb_of_sensors*action
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
        rate = rospy.Rate(8)  # 8 Hz
        
        while not rospy.is_shutdown():
            self.publish_goal(3,1)


            if self.navigating_to_goal:
                angle_to_goal = math.atan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x)
                angular = angle_to_goal - math.atan2(math.sin(angle_to_goal - self.robot_yaw), math.cos(angle_to_goal - self.robot_yaw))

                action, _states = self.model.predict(self.state)
                local_goal = self.bug.predict(self.state)
                action = self.DWA.predict(self.state, local_goal)

                #convert to linear and angular commands
                [linear, angular] = self.action2velocity(action)

                self.publish_velocity(linear, angular)  # P-controller for angular velocity
            else:
                self.publish_velocity(0.0, 0.0)

            # Check if robot has reached the goal
            if self.navigating_to_goal:
                distance_to_goal = math.sqrt((self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2)
                if distance_to_goal <= self.goal_reached_distance:
                    self.navigating_to_goal = False
                    self.publish_velocity(0.0, 0.0)  # Stop the robot
            
            rate.sleep()
            

if __name__ == '__main__':
    try:
        lidar_goal_generator = LidarGoalGenerator()
        lidar_goal_generator.run()
    except rospy.ROSInterruptException:
        pass
