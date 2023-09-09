import math
import numpy as np
import settings
import random
import time


class tangent_bug():
    '''
    implementation of a tangent bug algorithm for path planning with obstacle avoidance.
    '''

    def __init__(self):
        self.arc = 2*math.pi/settings.number_of_sensors #rad
        self.d_leave = 150
        self.d_min = 149
        self.following_boundary = False
        self.following_boundary_counter=0
        self.done =False
        self.min_dist = 150
        self.max_dist = 10
        
        self.foundPathCounter = 0
        self.tangent_direction = 1
        self.tangent_counter = 0
        self.normalize = True

        # PID Constants
        self.setpoint = 1.4  # Setpoint distance in meters should be the same as dwa?
        self.kp = 0.6  # Proportional gain
        self.ki = 0.05  # Integral gain
        self.kd = 0.0  # Derivative gain

        # PID Variables
        self.last_error = 0.0
        self.integral_error = 0.0
        self.sample_time = settings.mv_fw_dur  # PID loop sample time in seconds

        #short-term memory
        self.objects_last_updated =  [0]*settings.number_of_sensors
        self.previous_obs = [3]*(settings.number_of_sensors+6)


    def predict(self, obs):
        obs = obs[0][0] #flattening the list
        if self.normalize:
            obs[6:settings.number_of_sensors+6] = 100**obs[6:settings.number_of_sensors+6] #reconverting from normalized to real values
            obs[settings.number_of_sensors+6:] = obs[settings.number_of_sensors+6:]*np.pi
            obs[1] = 100**obs[1]
            obs[0] = obs[0]*np.pi #rad
            obs[2] = obs[2]*(settings.base_speed*20.0) #reconverting from normalized values
            obs[3] = obs[3]*np.pi
            obs[4:6] = obs[4:6]*50.0 

        sensors = obs[6:settings.number_of_sensors+6]

        goal_angle = obs[0]
        goal_distance = obs[1]
        vel_angle = obs[3]
        vel_norm = obs[2]

        #print(f"observation: {obs}")

        #angles =  obs[settings.number_of_sensors+6:] #angles are not included for now
        angles = np.linspace(-np.pi, np.pi-self.arc, settings.number_of_sensors)
        objects =[]
        orientations = []
        #create objects list to evaluate obstacles positions, and replace missing values with old observations.
        #values over 99 are the sensors that are "removed" by the RL agent
        #any distance greater than the treshold will be ceiled.

        #print(f"previous obs: {self.previous_obs}")
        #print(f"sensors: {sensors}")

        for i, sensor in enumerate(sensors):
            if sensor >= 10 and self.objects_last_updated[i] < 5:
                sensors[i] = self.previous_obs[i]
                self.objects_last_updated[i] += 1
            else:
                self.objects_last_updated[i] = 0
            objects.append(min(sensors[i], self.max_dist))
            orientations.append(angles[i])

        segments = self.compute_segments(objects)

        #print(f"sensors: {np.round(sensors,1)}")
        #print(f"updated sensors: {np.round(sensors,1)}")
        print(f"bug objects: {np.round(objects,1)}")
        #print(f"last updated: {self.objects_last_updated}")
        #print(f"bug distances: {np.round(objects,1)}")
        #print(f"segments: {segments}")
        print_angles = [x*180/math.pi for x in orientations]

        #fill narrow gaps where the drone couldn't safely pass
        objects = self.fill_gaps(objects, segments, orientations)

        
        if self.done: #finished episode, reset distances and PID state
            self.d_leave = 150
            self.d_min = 149
            self.min_dist = 150
            self.following_boundary_counter = 0
            self.following_boundary = False
            self.foundPathCounter = 0
            self.tangent_direction = 1
            self.tangent_counter = 0
            # PID Variables
            self.last_error = 0.0
            self.integral_error = 0.0
            #short-term memory
            self.objects_last_updated =  [0]*settings.number_of_sensors
            self.previous_obs = [3]*(settings.number_of_sensors+6)

        #find direction that minimizes distance to goal
        foundPath = False
        min_heuristic = 150000000
        best_idx=0
        for i, object in enumerate(objects):
            heuristic = self.compute_heuristic(object, orientations[i], goal_distance, goal_angle)
            if heuristic <= min_heuristic:
                min_heuristic = heuristic
                direction = orientations[i]
                best_idx = i
        #print(f"previous heuristic: {self.min_dist}")
        #print(f"new heuristic: {min_heuristic}")
        if min_heuristic <= self.min_dist-0.01:
            self.min_dist = min_heuristic
            foundPath = True
            self.foundPathCounter = 0
        heuristic = self.compute_heuristic(objects[best_idx], orientations[best_idx], goal_distance, goal_angle)
        

        if foundPath == False and self.following_boundary == False:
            direction = math.pi/2 - goal_angle
            print(f"best_idx angle: {orientations[best_idx]}")
            print(f"direction: {direction}")
            if objects[best_idx] < goal_distance:
                self.foundPathCounter += 1
                print("heuristic increased, go straight into goal, counter increased for boundary following")
            else:
                print("heuristic increased, but no obstacle blocking goal, going straight into goal.")
            goal = [goal_distance*math.cos(direction), goal_distance*math.sin(direction)]

        #if the heuristic didn't decrease after last couple actions, we need to enter into boundary following
        if self.foundPathCounter >= 4 and not self.following_boundary:
            print("entering boundary following")
            self.following_boundary = True
            self.following_boundary_counter=0
            self.tangent_counter = 0



        if(not self.following_boundary):
            
            #action = 12 - direction_idx + 32
            
            #print(f"direction: {np.round(direction*180/math.pi,2)}")
            # if goal_distance > objects[best_idx]: #why not just try to go past?
            #     goal = [objects[best_idx]*math.cos(direction), objects[best_idx]*math.sin(direction)]  #drone body frame ref
            # else:
            #     goal = [goal_distance*math.cos(direction), goal_distance*math.sin(direction)]  #drone body frame ref
            #take moving action
            goal = [goal_distance*math.cos(direction), goal_distance*math.sin(direction)]  #drone body frame ref

        else:


            closest_obstacle_idx = np.argmin(objects)
            tangent = orientations[closest_obstacle_idx]+math.pi/2*self.tangent_direction

            print(f"closest obstacle is at angle {orientations[closest_obstacle_idx]*180/math.pi} and distance {objects[closest_obstacle_idx]}. Tangent:  {tangent*180/math.pi}")
            command = self.calculate_pid_command(objects[closest_obstacle_idx])
            #print(f"PID command: {command}")
            tangent += command*self.tangent_direction
            #print(f"corrected tangent: {tangent*180/math.pi}")
            goal = [settings.mv_fw_spd_1*math.cos(tangent), settings.mv_fw_spd_1*math.sin(tangent)]

            object_to_avoid = segments[closest_obstacle_idx]
            #print(f"avoiding segment no. : {object_to_avoid}")
            self.d_leave, direction, idx = self.compute_d_leave(objects, orientations, goal_distance, goal_angle, segments)
            self.d_min, d_min_direction = self.compute_d_min(objects, orientations, goal_distance, goal_angle, object_to_avoid, segments)
            #print(f"d_leave: {self.d_leave}  d_min: {self.d_min}")
            #print(f"d_leave direction: {direction*180/math.pi}, d_min direction: {d_min_direction*180/math.pi}")
            #print(f"boundary folling counter: {self.following_boundary_counter}")
            #print(f"tangent counter: {self.tangent_counter}")

            #check if goal reached or escape found, or far from any obstacles
            if (self.done or self.d_leave < self.d_min*0.98 or min(objects) > 3.0):
                self.following_boundary_counter += 1
                if goal_distance > objects[idx]: #check if this gets the dwa stuck
                    goal = [objects[idx]*math.cos(direction), objects[idx]*math.sin(direction)]  #drone body frame ref
                else:
                    goal = [goal_distance*math.cos(direction), goal_distance*math.sin(direction)]  #drone body frame ref
                #print(f"done: {self.done}, closest object: {min(objects)}, d_leave/d_min ratio: {self.d_leave/self.d_min}")
                if self.following_boundary_counter > 4:
                    self.following_boundary = False
                    self.foundPathCounter = 0
                    print("switched back to normal path")

            else:
                self.following_boundary_counter = 0
                #if we've been following the boundary for a long time, try returning to normal state and switching tangent directions
                self.tangent_counter +=1
                if self.tangent_counter > 20:
                    self.tangent_counter = -20
                    self.tangent_direction = -1*self.tangent_direction
                    print("switching direction")
                    self.following_boundary = False
                    self.foundPathCounter = 0
                    self.d_leave = 150
                    self.d_min = 149
                    self.min_dist = 150
                    self.tangent_counter = 0
                    print("reset to normal behavior")
         
            


        self.previous_obs = objects
        #print(f"goal objects after method: {objects}")
        #print(f"goal distance: {goal_distance} angle: {goal_angle*180/math.pi}")
        #print(f"goal (conventional coordinates): {goal}")

        return goal




    def compute_d_leave(self, objects, angles, goal_dist, goal_angle, segments):


        distMin = 150
        for i, object in enumerate(objects):
                x_obj = object*math.cos(angles[i]+self.arc/2)
                y_obj = object*math.sin(angles[i]+self.arc/2)

                x_goal = goal_dist*math.sin(goal_angle) #reference frame for angle to goal is inverted
                y_goal = goal_dist*math.cos(goal_angle)

                dist_obj2goal = np.sqrt((y_goal-y_obj)**2 + (x_goal-x_obj)**2)

                #check if the dwa will be able to reach that point easily.
                if object < 10: #1. if the d_leave distance found is towards an obstacle, the real achievable distance is dist_obj2goal + the safety margin of the dwa.
                    dist_obj2goal += 1.5


                if dist_obj2goal < distMin:
                    distMin = dist_obj2goal
                    orientation = angles[i]
                    idx = i

        #print(f"orientation of d_leave: {orientation*180/math.pi}")

        return distMin, orientation, idx


    def compute_d_min(self, objects, angles, goal_dist, goal_angle, object_to_avoid, segments):
        distMin = self.d_min
        orientation = 1000.0
        for i, object in enumerate(objects):
            if segments[i] == object_to_avoid: #only update d_min if we confirmed this is on the boundary of the obstacle
                x_obj = object*math.cos(angles[i]+self.arc/2)
                y_obj = object*math.sin(angles[i]+self.arc/2)

                x_goal = goal_dist*math.sin(goal_angle) #reference frame for angle to goal is inverted
                y_goal = goal_dist*math.cos(goal_angle)

                dist_obj2goal = np.sqrt((y_goal-y_obj)**2 + (x_goal-x_obj)**2)

                if dist_obj2goal < distMin:
                    distMin = dist_obj2goal
                    orientation = angles[i]

        return distMin, orientation

    def compute_heuristic(self, object_dist, object_angle, goal_dist, goal_angle, verbose=False):

        x_goal = goal_dist*math.sin(goal_angle) #reference frame for angle to goal is inverted
        y_goal = goal_dist*math.cos(goal_angle)

        if object_dist < goal_dist:
            if verbose:
                print("object in front of goal!")
            x_obj = object_dist*math.cos(object_angle+self.arc/2)
            y_obj = object_dist*math.sin(object_angle+self.arc/2)

            dist_uav2obj = object_dist
            dist_obj2goal = np.sqrt((y_goal-y_obj)**2 + (x_goal-x_obj)**2)

            #print(f"angle considered: {object_angle*180/math.pi}")
            #print(f"object distance: {np.round(dist_uav2obj,2)}, obj to goal distance: {np.round(dist_obj2goal,2)} heuristic: {np.round(dist_uav2obj + dist_obj2goal,2)}")

            return max(0.1, dist_uav2obj + dist_obj2goal)
        else: #goal is in front of obstacle, heuristic is distance to goal after moving a bit into object_angle direction (0.1m/s)
            if verbose:
                print("goal in front of object!")
            x_obj = 0.1*math.cos(object_angle+self.arc/2)
            y_obj = 0.1*math.sin(object_angle+self.arc/2)

            dist_obj2goal = np.sqrt((y_goal-y_obj)**2 + (x_goal-x_obj)**2)
            if verbose:
                print(f"dist_obj2goal + 0.1 m: {dist_obj2goal + 0.1}")
                print(f"goal_dist: {goal_dist}")

            #print(f"Heuristic: {dist_obj2goal}")
            return max(0.1, dist_obj2goal + 0.1)


    def compute_segments(self, objects):
        """receives lidar data, and divides data into continuous segments. returns a list of the tags for each object.
        ex: for objects [1 1.1 1.2 1.3 5 5.1 3.1 3.2 3.1 0.9], will output [0 0 0 0 1 1 2 2 2 0] """

        discontinuity_treshold = 1.1
        ratio_threshold = 1.25

        ratios = []
        diff = []
        diff.append(objects[0]-objects[-1])
        ratios.append(objects[0]/objects[-1])
        for i in range(len(objects)-1):
            diff.append(objects[i+1]-objects[i])
            ratios.append(objects[i+1]/objects[i])

        segments = [0]*len(objects)
        discontinuities = 0
        for i, delta in enumerate(diff):
            #there is a discontinuity if there is a difference of at least 0.5m and at least 30% of the sensor mesurement
            if abs(delta) > discontinuity_treshold and (ratios[i] > ratio_threshold or 1/ratios[i] > ratio_threshold):
                discontinuities+=1
            segments[i] = discontinuities

        #completing the loop (if end and beginning of circle is the same segment)
        if abs(diff[0]) < discontinuity_treshold:
            for i, segment in enumerate(segments):
                if segment == discontinuities:
                    segments[i] = 0

        
        return segments

    def find_discontinuities(self, segments):
        """ receives an array of segments and computes where the discontinuities are.
        Returns a list with the index of the right item where there is a discontinuity ([0] between last and first element)"""
        discontinuities = []
        for i, id in enumerate(segments):
            left = segments[i-1]
            right = segments[i]
            if left != right:
                discontinuities.append(i)

        return discontinuities

    def fill_gaps(self, objects, segments, angles):
        #if the bug sees a escape window between two obstacles, but it is too narrow for the dwa to enter it, it will fail to avoid the local minimum.
        #we want to mark it as "obstacle"
        
        discontinuities = self.find_discontinuities(segments)
        temp_objects = objects.copy()

        #print(f"objects: {np.round(objects,1)}")
        #print(f"segment: {segments}")
        #print(f"discontinuities: {discontinuities}")

        corrected = False
        for idx, discontinuity in enumerate(discontinuities):
            if idx == len(discontinuities)-1:
                left = discontinuities[idx]-1 #get to the end of the previous discontinuity
                right = discontinuities[-1] #get to the start of the next discontinuity
            else:
                left = discontinuities[idx]-1
                right = discontinuities[idx+1]

            #process gaps
            #compute lenght of gap using al-kashi formula
            gap = np.sqrt(objects[left]**2 + objects[right]**2 -2*objects[left]*objects[right]*math.cos(angles[right]-angles[left]))
            #print(f"gap between {left} and {right}: {gap}")
            
            if gap < 0.4:
                if left < right:
                    for i in range(left, right):
                        temp_objects[i] = min(objects[left], objects[right])
                        #print(f"filled in gap for sensor {i}")
                else:
                    for i in range(right, left):
                        temp_objects[i] = min(objects[left], objects[right])
                        #print(f"filled in gap for sensor {i}")

                corrected = True
            
   
        if corrected:
            print(f"objects before: {np.round(objects,1)}")
            print(f"corrected objects: {np.round(temp_objects,1)}")

        objects[:] = temp_objects[:]

        return objects

    def calculate_pid_command(self, distance):

        # Calculate error and elapsed time
        error = self.setpoint - distance

        # Calculate proportional term
        proportional_term = self.kp * error

        # Calculate integral term
        self.integral_error += error * self.sample_time
        integral_term = self.ki * self.integral_error

        # Calculate derivative term
        derivative_error = (error - self.last_error) / self.sample_time
        derivative_term = self.kd * derivative_error

        # Calculate PID output
        pid_output = proportional_term + integral_term + derivative_term
        #print(f"error: {error} m")
        #print(f"unbounded pid: {pid_output}")
    
        # Bound PID output
        if pid_output < -0.5:
            pid_output = -0.5
        elif pid_output > 0.5:
            pid_output = 0.5

        # Store last error and elapsed time
        self.last_error = error
    

        return math.asin(pid_output)