import random
import os
import numpy as np
import settings
import math
import time
from bisect import bisect

from tangent_bug import tangent_bug


def normalize(obs):
    """ takes an observation (list) and outputs the normalized form"""
    #print(obs)
    obs[6:settings.number_of_sensors+6] = np.log(obs[6:settings.number_of_sensors+6]+0.0001)/np.log(100) #this way gives more range to the smaller distances (large distances are less important).
    #obs[settings.number_of_sensors+6:] = obs[settings.number_of_sensors+6:]*np.pi
    obs[1] = np.log(obs[1]+0.0001)/np.log(100)
    obs[0] = obs[0]/np.pi #rad
    obs[2] = obs[2]/(settings.base_speed*20.0) #reconverting from normalized values
    obs[3] = obs[3]/np.pi
    obs[4:6] = obs[4:6]/50.0 

    return obs



class gofai():
    '''
    naive implementation of a pursuing algorithm with obstacle avoidance.
    '''

    def __init__(self):
        self.arc = 2*math.pi/settings.number_of_sensors #rad
        self.heading_coeff = 1
        self.safety_coeff = 4
        self.safety_dist = 1.5
        self.previous_obs = [3]*(settings.number_of_sensors+6)
        self.bug = tangent_bug()




    def predict(self, obs, goal):
        '''
        observation is in the form [angle, d_goal, vel_norm, vel_angle, y_pos, x_pos, d1, d2, ..., dn] where d1 starts at 180 deg and goes ccw, velocities are in drone's body frame ref
        actions are distributed as following:
        0-15: small circle
        16-31: medium small circle
        32-47: medium big circle
        48-63: big circle
        '''
        obs = obs[0][0] #flattening the list

        
        #read goal from observation (when not using tangent bug)
        #goal_angle = obs[0]*math.pi #rad
        #global_goal_distance = obs[1]

        #read goal coordinates from tangent bug
        x_goal = goal[0]
        y_goal = goal[1]
        global_goal_distance = np.sqrt(x_goal**2 + y_goal**2)
        #print(f"received goal (relative): {[x_goal,y_goal]}")

        
        vel_angle = obs[3]
        vel_norm = obs[2]
        x_pos = obs[5]
        y_pos = obs[4]
        predicted_delay = settings.delay*5 #accouting for predicted latency, and simulation time vs real time
        x_offset = predicted_delay*vel_norm*math.cos(vel_angle)*1.25
        y_offset = predicted_delay*vel_norm*math.sin(vel_angle)*1.25

        sensors = obs[6:settings.number_of_sensors+6] 
        angles = np.arange(-np.pi, np.pi, self.arc)
        #print(f"sensors: {np.round(sensors,1)}")
        #print(f"angles: {angles}")

        # ---------------- random and greedy baselines -----------------------------
        algo = "A2C-B"
        if(algo == "GOFAI"):
            #chooses k closest sensors
            #k_sensors = 3
            #chosen_idx = np.argpartition(sensors, k_sensors)[:k_sensors]
            #sensor_output = np.ones(settings.number_of_points)*100
            #for idx in chosen_idx:
            #    sensor_output[idx] = sensors[idx]
            #sensors = sensor_output
            #randomly chooses a subset of sensors to process (imitating RL agent)
            n_sensors = 428
            chosens = random.sample(range(len(sensors)),k=(settings.number_of_points-n_sensors))
            #print(chosens)
            for idx in chosens:
                sensors[idx] = 100
        #print(f"sensors dwa: {np.round(sensors,1)}")
        # -----------------------------------------------------------------


        objects =[]
        orientations = []
        #create objects list to evaluate obstacles positions, and replace missing values with old observations
        #values over 99 are the sensors that are "removed" by the RL agent
        for i, sensor in enumerate(sensors):
            if sensor < 99:
                if sensor >= 66:
                    sensors[i] = self.previous_obs[i]
                objects.append(sensors[i])
                orientations.append(angles[i])

        x_objects = []
        y_objects = []
        for object,angle in zip(objects,orientations):
            x_objects.append(object*math.cos(angle+self.arc/2) - x_offset)
            y_objects.append(object*math.sin(angle+self.arc/2) - y_offset)
        x_objects = np.array(x_objects)
        y_objects = np.array(y_objects)
            
        
        #print(f"dwa objects: {np.round(objects,1)}")
        #print(orientations)
        #print(len(objects))
        if len(objects) == 0: #if there is no obstacles, go straight to the goal at max speed
            thetas =  np.linspace(-math.pi, math.pi, settings.action_discretization+1)
            goal_angle = math.atan2(y_goal,x_goal)
            direction = bisect(thetas, goal_angle)
            if (thetas[direction]-goal_angle > goal_angle - thetas[direction-1]): #find which discretized value is closest
                direction -= 1
            action = (16 - direction%settings.action_discretization + round(0.75*settings.action_discretization))%settings.action_discretization  #transform the airsim action space (starts at 90 deg and goes cw)
            return action + 48
        
        #sensors = np.concatenate((sensors,sensors)) #this way we can more easily slice the angles we want
        #angles = np.concatenate((angles,angles))
        bestBenefit = -1000
        action = 0
        angle_increment = 2*math.pi/settings.action_discretization
        for i in range(settings.action_discretization*4): #4 velocities time 16 directions
            theta = math.pi/2 - angle_increment*(i%settings.action_discretization)  #in the action space, the circle starts at 90 deg and goes cw (drone body frame reference)

            #computing new distance to goal
            travel_speed = min(2, settings.base_speed*3**(i//settings.action_discretization)) #travelling speed can be 0.1, 0.3, 0.9, or 2 m/s 
            x_dest = travel_speed*math.cos(theta)*0.4*(settings.mv_fw_dur+predicted_delay*0.25) + vel_norm*math.cos(vel_angle)*(0.75+predicted_delay*1.25) # correcting for current speed since change in speed isn't instantaneous
            y_dest = travel_speed*math.sin(theta)*0.4*(settings.mv_fw_dur+predicted_delay*0.25) + vel_norm*math.sin(vel_angle)*(0.75+predicted_delay*1.25)

            new_dist = np.sqrt((x_goal-x_dest)**2+(y_goal-y_dest)**2)
            #computing the closest obstacle to the trajectory
            minDist = self.safety_dist
            if (len(objects) > 0):
                dist = self.shortest_distance_on_trajectory(x_objects,y_objects,x_dest,y_dest)
                if dist < minDist:
                    minDist = dist

            #computing the benefit
            benefit = self.heading_coeff*(global_goal_distance-new_dist) - self.safety_coeff*(self.safety_dist - minDist)
            #print(f"heading term: {global_goal_distance-new_dist}")
            #print(f"safety term: {self.safety_dist - minDist}")
            if benefit > bestBenefit:
                bestBenefit = benefit
                mindistAction = minDist
                headingTerm = self.heading_coeff*(global_goal_distance-new_dist)
                safetyTerm = self.safety_coeff*(self.safety_dist - minDist)
                action =i
                direction = theta


        self.previous_obs = sensors
        #print(f"min predicted distance in chosen dwa action: {mindistAction}")
        #print(f"heading term: {headingTerm} safety term: {safetyTerm}")
        #print(f"full loop action: {action}")

        ### -----------printing info on the chosen action-------------------------------------------------------------
        travel_speed = min(2, settings.base_speed*3**(action//settings.action_discretization)) #travelling speed can be 0.5, 1, 2, or 4 
        x_dest = travel_speed*math.cos(direction)*0.4*(settings.mv_fw_dur+predicted_delay*0.5) + vel_norm*math.cos(vel_angle) * (0.75+predicted_delay)  + x_pos # correcting for current speed since change in speed isn't instantaneous
        y_dest = travel_speed*math.sin(direction)*0.4*(settings.mv_fw_dur+predicted_delay*0.5)  + vel_norm*math.sin(vel_angle) * (0.75+predicted_delay) + y_pos
        #print(f"desired angle: {np.round(direction*180/math.pi,1)}")
        #print(f"current speed: {[np.round(vel_norm,1), np.round(vel_angle*180/np.pi,1)]}")
        #print(f"min distance in chosen trajectory: {np.round(minDist,5)}")
        #print(f"objects: {np.round(objects,1)}")
        #print(f"orientations: {np.round(orientations,2)}")
        #print(f"sensors: {np.round(sensors,1)}")
        #print(f"goal_distance: {global_goal_distance} angle: {goal_angle*180/math.pi}")
        #print(f"destination: {[np.round(y_dest,1), np.round(x_dest,1)]}")
        #print(f"destination: {np.round(now,2)}")
        #print(f"min distance in chosen trajectory: {minDist}")
        #print(f"goal speed: {travel_speed}")
        #print(f"received speed: {np.round(np.sqrt(x_vel**2 + y_vel**2),2)}")
        #print(f"received pos: {[np.round(y_pos,2), np.round(x_pos,2)]}")
        #print(f"corrected pos: {[np.round(y_pos+y_offset,2), np.round(x_pos+x_offset,2)]}")
        #print(f"predicted destination: {[np.round(y_dest,2), np.round(x_dest,2)]}")
        #print(f"new_dist: {new_dist}")
        #---------------------------------------------

        
        return action

    def shortest_distance_on_trajectory(self, X, Y, x2, y2):
        """
        finds the closest point from the given points (X,Y) to the (x2,y2) line segment (from the origin). outputs the closest distance to that segment
        """
        norm = x2*x2 + y2*y2
        if norm == 0:
            return np.sqrt(np.min(X**2+Y**2))

        dotProducts = X*x2 + Y*y2

        params = np.ones(X.size)*-1
        params = np.clip(dotProducts/norm,0,1)

        xx = params*x2
        yy = params*y2

        dx = X - xx
        dy = Y - yy

        norms = dx**2 + dy**2
        minDist = np.sqrt(norms.min())

        return minDist
