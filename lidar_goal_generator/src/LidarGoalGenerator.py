#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math
import settings
from utils import gofai, cost_function
from tangent_bug import tangent_bug
import dynamic_window_approach as wheeled_dwa
from dynamic_window_approach import Config
from dynamic_window_approach import RobotType
import numpy as np
from bisect import bisect
import time
from os import path

#RL libraries
#import torch
#from model import PyTorchMlp

class LidarGoalGenerator:
    def __init__(self, filepath):
        rospy.init_node('lidar_goal_generator', anonymous=True)

        #RL agent setup
        self.nb_of_sensors = settings.number_of_sensors
        self.state = np.zeros((1, 1, 4 + 2 + settings.number_of_sensors)) #todo: check if it works with only a list
        #self.model = PyTorchMlp()
        #self.model.load_state_dict(torch.load('torch_A2C_model.pt')) #put the model in the working directory
        #self.model.eval()
        self.RL = False #switch between pytorch and cost function
        #self.model = A2C.load(filepath)
        self.DWA = gofai()
        self.bug = tangent_bug()
        self.config = Config()
        
        # Parameters
        self.goal_reached_distance = 0.18  # Distance threshold to consider the goal reached
        self.linear_speed = 0.05  # Linear speed for moving towards the goal
        self.angular_speed = 1.1
        
        # Goal coordinates
        self.goals = [[2.0, 0.0], [0.0, 0.0]]
        self.goal_x = 2.0
        self.goal_y = 0.0
        
        # Current robot pose
        self.old_x = 0.0
        self.old_y = 0.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        
        # Flag to indicate if a goal is being navigated to
        self.navigating_to_goal = True
        self.current_goal_idx = 0

        #metrics
        self.processing_times = []
        self.CPU_processing_times = []
        self.mission_time = 0
        self.mission_start = time.time()
        self.distance_traveled = 0
        self.observations = []
        self.actions = []

        # LiDAR subscriber
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        #LiDAR objects
        self.x_objects = np.ones(self.nb_of_sensors)*10
        self.y_objects = np.ones(self.nb_of_sensors)*10
        self.unseen_idx = []
        
        # Robot pose subscriber
        self.pose_sub = rospy.Subscriber('/odom', Odometry, self.pose_callback)

        # Robot goal subscriber
        #self.goal_sub = rospy.Subscriber('/goal', Odometry, self.goal_callback)
        
        # Velocity publisher
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # goal publisher
        self.goal_pub = rospy.Publisher('/goal', Odometry, queue_size=10)

        
    def lidar_callback(self, scan_data):
        # Process LiDAR data for RL agent: update seen sensors with new values, and try to infer other sensors from odom and old values
        # The reference frame for the lidar is: x axis in front of robot (angle 0 in front, positive angle towards the left)
        
        distances = np.array(scan_data.ranges)
        indexes = np.arange(0,distances.size)
        angles = scan_data.angle_min + indexes * scan_data.angle_increment

        #look at the min and max angle (it should be -np.pi to np.pi)
        #add np.pi to get between o and 2pi
        #add the robot_yaw, and loop back to the beginning with a modulo
        angles = angles + np.pi + self.robot_yaw
        angles = (angles*180/np.pi) % 360
        angles = angles*np.pi/180 - np.pi

        #removing zeros from lidar data
        initial_size = distances.size
        angles = angles[distances != 0.0]
        distances = distances[distances != 0.0]
        
        #print(f"lidar ratio of good/bad samples: {initial_size/distances.size}")
        # print(f"LiDAR distances: {distances[0:5]}")
        # print(f"LiDAR distances: {distances[indexes.size-5:]}")
        # print(f"LiDAR angles: {angles[0:5]}")
        # print(f"LiDAR angles: {angles[indexes.size-5:]}")
        
        left_angle=-np.pi#keep the same convention as the agent learned, even tho the lidar won't fill the whole circle.
        right_angle=np.pi

        lidar_FOV =  math.ceil(scan_data.angle_max - scan_data.angle_min)

        #spliting both new points and old objects into ranges of angles
        theta = 2*np.pi/self.nb_of_sensors
        for i in range(self.nb_of_sensors):
            if i == 0:
                thetas = [left_angle]
            else:
                thetas.append(thetas[i-1]+theta)

        if len(angles) == 0: #lidar not seeing anything!
            return np.ones(self.nb_of_sensors)

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
            if len(distances_by_sensor[i]) != 0: #missing lidar values!
                sensors[i] = min(distances_by_sensor[i])
            else:
                sensors[i] = 66 #with no information, set to 66, which will be ignored
        #print(f"final sensors: {np.round(sensors,2)}")
        #wait = input()

        #normalize and write processed data to state
        normalized_sensors = [0 for x in range(self.nb_of_sensors)]
        for i in range(self.nb_of_sensors):
            normalized_sensors[i] = np.log(sensors[i]+0.00001)/np.log(100) #this way gives more range to the smaller distances (large distances are less important).
            normalized_sensors[i] = min(1.0,max(-1.0,normalized_sensors[i]))
        
        self.state[0][0][6:self.nb_of_sensors+6] = normalized_sensors
        #print(f"state: {np.round(self.state,2)}")

        
    def pose_callback(self, odom_data):
        #update distance traveled
        delta_x = odom_data.pose.pose.position.x - self.robot_x
        delta_y = odom_data.pose.pose.position.y - self.robot_y
        self.distance_traveled += np.linalg.norm([delta_x, delta_y])
        # Get current robot pose
        self.robot_x = odom_data.pose.pose.position.x
        self.robot_y = odom_data.pose.pose.position.y
        
        self.robot_vx = odom_data.twist.twist.linear.x
        self.robot_vy = odom_data.twist.twist.linear.y

        orientation_q = odom_data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, self.robot_yaw ) = euler_from_quaternion (orientation_list)

        #normalize and write robot pose and velocity to state
        vel_norm = min(np.sqrt(self.robot_vx**2+self.robot_vy**2)/(settings.base_speed*20),1) #max speed is 20*base speed (2m/s)
        vel_angle = math.atan2(self.robot_vy,self.robot_vy)/math.pi
        self.state[0][0][2:4] = [vel_norm, vel_angle]
        self.state[0][0][4:6] = [self.robot_x/50, self.robot_y/50]
          
          
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

        angle_to_goal = math.atan2(self.goal_x - self.robot_x, self.goal_y - self.robot_y) # angle to goal need to be from y axis and in clock wise direction (airsim convention)
        distance_to_goal = math.sqrt((self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2)

        #correct with current orientation
        #angle_to_goal = angle_to_goal - self.robot_yaw

        #normalize and write robot pose to state directly
        print(f"relative goal: [theta,dist] = {[np.round((np.pi/2-angle_to_goal-self.robot_yaw)*180/np.pi,1), np.round(distance_to_goal,2)]}")
        distance_to_goal = np.log10(distance_to_goal+0.00001)/np.log10(100) #this way gives more range to the smaller distances (large distances are less important).
        distance_to_goal = min(1,max(-1,distance_to_goal))
        angle_to_goal = angle_to_goal/np.pi #since it is already between [-180,180] and we want a linear transformation.
        self.state[0][0][0:2] = [angle_to_goal, distance_to_goal]

        self.goal_pub.publish(goal_msg)


    def action2velocity(self, action):
        """
        discretization of the position actions: they are placed in a circle around the UAV. this circle is divided by 16 for the directions.
        there are 4 circles with different radius depending on how far the drone wants to go. the direction start in the front of the robot and go clockwise.
        """
        duration = 0.125
        speed = self.linear_speed

        if action < settings.action_discretization * 1:
            #short distance
            speed *= 1
        elif action < settings.action_discretization * 2:
            #medium short dist
            speed *= 3
        elif action < settings.action_discretization * 3:
            #medium long
            speed *= 9
        else:
            print("Wrong action index!")

        action = action % settings.action_discretization
        angle = np.pi/2-2*np.pi/settings.action_discretization*action #the negative sign is to go clockwise
        print(f"dwa heading: {angle*180/np.pi}")
        #correcting for current yaw
        angle = angle - self.robot_yaw
        if angle > np.pi:
            angle = np.pi - angle
        print(f"estimated yaw: {np.round(self.robot_yaw*180/np.pi,1)}")
        #print(f"wanted angle corrected for yaw: {np.round(angle*180/np.pi,1)}")
        v_front =  speed*math.cos(angle)
        v_side = speed*math.sin(angle) #vy should be close to 0, if not, rotate:
        v_angular = self.angular_speed*math.sin(angle)
        #print(f"wanted v_front: {v_front}")
        #print(f"wanted v_side: {v_side}")
        if v_front < -0.1: #do not allow high backwards velocities, turn around instead
            linear = 0
            if math.sin(angle) > 0:
                angular = self.angular_speed*1.8 #rotate faster
            else:
                angular = -self.angular_speed*1.8
            print(f"rotating to go backwards")
        elif abs(angle) > np.pi/4: #if we are badly aligned with the goal, rotate and go slower
            linear = v_front*0.25
            angular = v_angular*1.5
            print(f"aligning with the goal! need to rotate {np.round(angle*180/np.pi,1)} deg")
        elif abs(angle) > np.pi/6: #if we are badly aligned with the goal, rotate and go slower
            linear = v_front*0.45
            angular = v_angular*1.3
            print(f"aligning with the goal! need to rotate {np.round(angle*180/np.pi,1)} deg")
        elif abs(angle) > np.pi/9: #if we are badly aligned with the goal, rotate and go slower
            linear = v_front*0.7
            angular = v_angular*1.2
            print(f"aligning with the goal! need to rotate {np.round(angle*180/np.pi,1)} deg")
        else:
            angular = v_angular
            linear = v_front

        return [linear, angular]
    
    def apply_mask(self, state, chosen_sectors):
        """
        takes the full state and action as input, and returns a partial observation based on the chosen action
        """
        #print(f"full state: {np.round(state,2)}")
        obs = state.copy()
        if self.RL: #convert the tensor output into an array
            action = chosen_sectors.detach().numpy().flatten()
            action = np.round((action+1)/2.0) #converting to 0s and 1s (temporary)
        else: #cost function
            action = chosen_sectors
        print(f"mask action: {action}")
        sensors = obs[0][0][6:settings.number_of_sensors+6]

        #print(f"action: {action}")
        #print(f"wanted sensors: {np.sum(action)}")
        #find the k highest sensors, then save them for the dwa algorithm
        chosen_idx = np.argpartition(action, -settings.k_sensors)[-settings.k_sensors:]
        sensor_output = np.ones(settings.number_of_sensors)*100
        for idx in chosen_idx:
            sensor_score = action[idx]
            if (sensor_score >= 0.5):
                sensor_output[idx] = sensors[idx]
        #for i, sensor in enumerate(sensors):
        #    if sensor < 2.5:
        #        sensor_output[i] = sensors[i]
        #closest = np.argmin(sensors)
        #sensor_output[closest] = 100

        obs[0][0][6:settings.number_of_sensors+6] = sensor_output

        #print(f" state: {state}")
        #print(f"outputed obs: {obs}")

        return obs
    
    def update_obstacles(self, observation):
        #takes in the measured lidar values and updates unseen sector with the previous positions of obstacles

        state = observation.copy()
        measured_sensors = 100**state[0][0][6:self.nb_of_sensors+6]
        self.unseen_idx = [1 if sensor >= 66 else 0 for sensor in measured_sensors]
        print(f"unseen_idx: {self.unseen_idx}")
        dx = self.robot_x - self.old_x
        dy = self.robot_y - self.old_y

        self.old_x = self.robot_x
        self.old_y = self.robot_y 
        print(f"dx: {np.round(dx,2)} dy: {np.round(dy,2)}")

        theta = 2*np.pi/self.nb_of_sensors
        for i in range(self.nb_of_sensors):
            if i == 0:
                thetas = [-np.pi]
            else:
                thetas.append(thetas[i-1]+theta)

        #1) update obstacles positions
        for i in range(self.nb_of_sensors):
            self.x_objects[i] -= dx
            self.y_objects[i] -= dy

        #2) use obstacles positions to update sensors
        obstacles_by_sensor = [[] for i in range(self.nb_of_sensors)]

        object_angles = np.arctan2(self.y_objects, self.x_objects)

        object_distances = np.sqrt(self.x_objects**2+self.y_objects**2)
        #print(f"object angles: {np.round(object_angles*180/np.pi,1)}")
        print("                                     ")
        print(f"measured sensors: {np.round(measured_sensors,2)}")
        print(f"object_distances: {np.round(object_distances,2)}")
        print("                                     ")

        for i, object_angle in enumerate(object_angles):
            ith_sensor = bisect(thetas,object_angle)
            obstacles_by_sensor[ith_sensor-1].append(object_distances[i])

        object_sensors = [0 for x in range(self.nb_of_sensors)]   
        for i in range(self.nb_of_sensors):
            if len(obstacles_by_sensor[i]) != 0:
                object_sensors[i] = min(obstacles_by_sensor[i])
                #print(f"updated unseen sensor! angle: {np.round(thetas[i]*180/np.pi,1)} new dist: {np.round(sensors[i],2)}")
            else:
                object_sensors[i] = 66
        print(f"obstacle sensors: {np.round(object_sensors,2)}")

        #3) update sensors and seen objects with ground truth
        for i in range(self.nb_of_sensors):
            if measured_sensors[i] >= 66:
                measured_sensors[i] = object_sensors[i]
            else:
                self.x_objects[i] = measured_sensors[i]*math.cos(thetas[i])
                self.y_objects[i] = measured_sensors[i]*math.sin(thetas[i])
        print(f"final sensors: {np.round(measured_sensors,2)}")
        #normalize back
        normalized_sensors = [0 for x in range(self.nb_of_sensors)]
        for i in range(self.nb_of_sensors):
            normalized_sensors[i] = np.log(measured_sensors[i]+0.00001)/np.log(100)
            normalized_sensors[i] = min(1.0,max(-1.0,normalized_sensors[i]))

        state[0][0][6:self.nb_of_sensors+6] = normalized_sensors.copy()

        return state


        
    def run(self):
        rate = rospy.Rate(8)  # 8 Hz
        initialized = False
        
        while not rospy.is_shutdown():
            self.publish_goal(1,0.5)

            start = time.perf_counter()
            startCPU = time.process_time_ns()
            if self.navigating_to_goal and initialized: # we want to skip the first iteration as the subscribers haven't yet read data
                
                print(f"Robot pose: [x,y] = {[np.round(self.robot_x,2), np.round(self.robot_y,2)]}")
                print(f"Goal [x,y]: {[np.round(self.goal_x,2), np.round(self.goal_y,2)]}")
                #save state in another variable so that it doesn't get overwritten by the subscriber mid-process
                observation = self.state.copy()
                observation = self.update_obstacles(observation)
                #print(f"copied observation: {np.round(observation,2)}")
                self.observations.append(observation)

                if self.RL:
                    chosen_sectors = self.model(torch.from_numpy(observation).float()) #inference profiling
                else:
                    chosen_sectors = cost_function(observation) #performance profiling (closer to real agent behavior)
                    
                self.actions.append(chosen_sectors)
                print("-----------------bug start ----------------")
                local_goal = self.bug.predict(observation)
                print(f"bug local goal [x,y]: {[np.round(local_goal[0],2), np.round(local_goal[1],2)]}")
                print("--------------- bug end -------------------")
                observation[self.unseen_idx==1] *=1.5 #give less importance to virtual objects for dwa (smaller margin)
                #observation = self.apply_mask(observation, chosen_sectors)
                start = time.perf_counter()
                action = self.DWA.predict(observation, local_goal)
                [linear, angular] = self.action2velocity(action) #convert to linear and angular commands
                print(f"angular vel: {np.round(angular,2)} linear vel: {np.round(linear,2)}")
                mid = time.perf_counter()
                #velocitiy_commands = DWA(self.state, local_goal, self.config, self.robot_yaw)
                #print(f"angular vel: {velocitiy_commands[1]} linear vel: {velocitiy_commands[0]} for wheeled dwa")
                end = time.perf_counter()
                #print(f"dwa process time: {np.round(mid-start,3)}")
                
                print("-------------------------------------------------")
                

                self.publish_velocity(linear, angular)  # P-controller for angular velocity
            else:
                self.publish_velocity(0.0, 0.0)
            end = time.perf_counter()
            endCPU = time.process_time_ns()
            #print(f"processing time: {(end-start)*1000} ms")
            #print(f"CPU processing time: {(endCPU-startCPU)/1000} muS")
            self.processing_times.append(end-start)
            self.CPU_processing_times.append(endCPU-startCPU)
            initialized=True

            #waiting for human input to take another step
            debug = input()

            # Check if robot has reached the goal
            if self.navigating_to_goal:
                distance_to_goal = math.sqrt((self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2)
                if distance_to_goal <= self.goal_reached_distance:
                    print("                                                     ")
                    print("===================REACHED GOAL======================")
                    print("                                                     ")
                    self.bug.done=True
                    self.current_goal_idx = 1 - self.current_goal_idx
                    self.publish_velocity(0.0, 0.0)  # Stop the robot
                    self.mission_time = time.time() - self.mission_start

                    self.goal_x = self.goals[self.current_goal_idx][0]
                    self.goal_y = self.goals[self.current_goal_idx][1]
                    docs_dir=path.expanduser('~/Documents')
                    with open(path.join(docs_dir, 'logging', 'observations'), "wb") as f:
                        np.save(f,np.array(self.observations),allow_pickle=True)
                    with open(path.join(docs_dir, 'logging', 'actions'), "wb") as f:
                        np.save(f,np.array(self.actions),allow_pickle=True)
                else:
                    self.bug.done=False
            
            rate.sleep()
          
def DWA(obs, goal, config, yaw):
    obs = obs[0][0]
    # goal position [x(m), y(m)]
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([obs[4], obs[5], yaw, obs[2], obs[3]])
    angles = np.linspace(-np.pi, np.pi-np.pi/60, 60)
    obstacles = obs[6:]
    ob = []
    for obstacle, angle in zip(obstacles,angles):
        if obstacle < 100:
            ob_x = obstacle*math.cos(angle)
            ob_y = obstacle*math.sin(angle)
            ob.append([ob_x, ob_y])
    ob = np.array(ob)
    if ob.shape[0] == 1:
        ob = np.reshape(ob, (1, 2))
    u, predicted_trajectory = wheeled_dwa.dwa_control(x, config, goal, ob)
    return u

if __name__ == '__main__':
    try:
        lidar_goal_generator = LidarGoalGenerator('../model.pkl')
        lidar_goal_generator.run()
    except rospy.ROSInterruptException:
        pass
