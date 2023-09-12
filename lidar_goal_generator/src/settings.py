import os
import math
import random


# ---------------------------
# imports 
# ---------------------------
# Augmenting the sys.path with relavant folders
settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
proj_root_path = os.path.abspath(settings_dir_path + "/..")

# ---------------------------
# file handling
# ---------------------------
chk_p_name_style = "0.hf5"  # the checkpoint obj will create a file with this style
chk_p_name_style_baselines = "0.pkl"
max_chck_pt_per_zone = 5  # pay attention

logging_interval = 100
checkpoint_interval = 10000

# ---------------------------
# zoning
# ---------------------------
# how many zones for each variable for the entire range. Note that frequency
# of moving to a new zone is not determined here
zone_dic = {"Seed": 1, "NumberOfDynamicObjects": 1, "MinimumDistance": 1, "VelocityRange": 1, "End": 1}  # pay attention

# update_zone_success_threshold = 50
acceptable_success_rate_to_update_zone = 0.96  # after what ratio of success up the zone # pay attention
update_zone_window = 100  # the window within which the  update_zone_accpetable_success_rate
# needs to be achieved. Note that at the begining of every new window we zero out the achieved ratio


# ------------------------------------------------------------ 
#                               -space related-
# -----------------------------------------------------------
# ---------------------------
# range #pay attention
# ---------------------------
# TODO: set default to something besides easy or fix the number of Mutables equal
default_range_dic = easy_range_dic = {"End": zone_dic["End"] * ["Mutable"],
                                      "MinimumDistance": [2],
                                      "EnvType": ["Indoor"],
                                      "ArenaSize": [[50, 50, 20]],
                                      "PlayerStart": [[0, 0, 0]],
                                      "NumberOfDynamicObjects": list(range(0, 1)),
                                      "Walls1": [[255, 255, 10]],
                                      "Seed": list(range(0, 10000)),
                                      "VelocityRange": [[5, 25]],
                                      "Name": ["Name"],
                                      "NumberOfObjects": list(range(0, 1))}

medium_range_dic = {"End": zone_dic["End"] * ["Mutable"],
                    "MinimumDistance": [2],
                    "EnvType": ["Indoor"],
                    "ArenaSize": [[60, 60, 20]],
                    "PlayerStart": [[0, 0, 0]],
                    "NumberOfDynamicObjects": list(range(0, 1)),
                    "Walls1": [[255, 255, 10]],
                    "Seed": list(range(0, 5000)),
                    "VelocityRange": [[0, 3]],
                    "Name": ["Name"],
                    "NumberOfObjects": list(range(2,4))}

hard_range_dic = {"End": zone_dic["End"] * ["Mutable"],
                  "MinimumDistance": [3],
                  "EnvType": ["Indoor"],
                  "EnvType": ["Indoor"],
                  "ArenaSize": [[100, 100, 10]],
                  "PlayerStart": [[0, 0, 0]],
                  "NumberOfDynamicObjects": list(range(75, 76)), #hard seed: 100 obstacles
                  "Walls1": [[255, 255, 10]],
                  "Seed": list(range(1,2)), #hard seed: seed 0. random: list(range(0,1000))
                  "VelocityRange": [[0.0, 0.0]],
                  "Name": ["Name"],
                  "NumberOfObjects": list(range(24,25))}

difficulty = "hard" #choose between easy (or default), medium, and hard

# ------------------------------------------------------------
#                               -game related-
# ------------------------------------------------------------
game_proc_pid = ''  # process associa

# TODO: this has to infered.
max_zone = zone_dic["End"]  # should be equal to mutable or total number of zones possible
# ---------------------------
# sampling frequency
# ---------------------------
# game config variables
# environment_change_frequency = {"Seed":5, "NumberOfObjects":1,\
#        "NumberOfDynamicObjects": 20, "MinimumDistance": 30, "VelocityRange":40} #this is based on episodes

end_randomization_mode = "inclusive"  # whether each level of difficulty should be inclusive (including the previous level) or exclusive

# how frequently to update the environment this is based on epides
environment_change_frequency = {"Seed": 1, "NumberOfObjects": 10, "End": 1} #the keywords are the variables to randomize, and the int associated is the number of episodes before randomization (i._. 1 means every episode, etc.)

# ------------------------------------------------------------
#                               -Drone related-
## ------------------------------------------------------------
#ip = '10.243.49.243'
ip = '127.0.0.1'

#---------------------------
# PPO
#---------------------------

# drone controls
duration_ppo = 0.3
move_by_velocity = True
move_by_position = False
# ---------------------------
# DDPG parameters
# ---------------------------

episode_count_cap = 5000
batch_size = 64  # ToDo: Determine what this value is
gamma = 0.99
tau = 0.001  # Target Network HyperParameters
lra = 0.0001  # Learning rate for Actor
lrc = 0.001  # Lerning rate for Critic

# Drone controls
min_throttle = 0.7
duration = 1

explore = 100000.
epsilon = 1
# ---------------------------
# DQN parameters
# ---------------------------
double_dqn = False
policy = "shallow" #"shallow" or "deep"

# ---------------------------
# Discrete action space parameters
# ---------------------------
#actions durations and speeds
"""
action durations here correspond to simulated time, so if the clockspeed is 
set to 2 in the airsim settings, an action of 50 ms here will take 25 ms.

NOTE: the learning and other rl stuff takes about 10 ms per step with the A2C algorithm (for example).
So the real frequency is always lower than the minimal action duration.
Also, increasing the clockspeed virtually increases the latency of the agent
(ex. the 10 ms becomes 20 ms in simulation time).
however, a part of the latency is explained by a waiting time to collect the state, which is also sped up a little bit.
this means the "percieved" latency doesn't increase linearly with the clockspeed, but increases nonetheless, slower than linearly.
"""


# ---------------------------
# action space configuration
# ---------------------------
timedActions = False
positionActions = True
action_discretization = 16 #this needs to be a square number and greater than one if timedActions is set to true! 
number_of_sensors = 12
number_of_points = 12
k_sensors = number_of_sensors  #the maximum amount of sensors the agent can choose at any time
assert(action_discretization > 1)
assert(action_discretization%4 == 0) #make sure it is divisible by 4 so that polar transformations work
if timedActions:
    assert(int(math.sqrt(action_discretization) + 0.5) ** 2 == action_discretization)

base_speed = 0.05
mv_fw_dur = 0.125
rot_dur = 0.125
delay = 0.00001 #artificial latency
# yaw_rate = (180/180)*math.pi #in degree
mv_fw_spd_1 = 1
mv_fw_spd_2 = 2
mv_fw_spd_3 = 3
mv_fw_spd_4 = 4
mv_fw_spd_5 = 5
# yaw_rate = (180/180)*math.pi #in degree
yaw_rate_1_1 = 108.  # FOV of front camera
yaw_rate_1_2 = yaw_rate_1_1 * 0.5  # yaw right by this angle
yaw_rate_1_4 = yaw_rate_1_2 * 0.5
yaw_rate_1_8 = yaw_rate_1_4 * 0.5
yaw_rate_1_16 = yaw_rate_1_8 * 0.5
yaw_rate_2_1 = -216.  # -2 time the FOV of front camera
yaw_rate_2_2 = yaw_rate_2_1 * 0.5  # yaw left by this angle
yaw_rate_2_4 = yaw_rate_2_2 * 0.5
yaw_rate_2_8 = yaw_rate_2_4 * 0.5
yaw_rate_2_16 = yaw_rate_2_8 * 0.5


# ---------------------------
# back up params
# ---------------------------
bu_dir_default = os.path.join(proj_root_path, "data", "backup")  # use this as default back up dir
bu_dir = bu_dir_default
# bu_dir = "E:\\backup"
backup_folder_name_style = "bu_0"  # the backup obj will create a file with this style

# ---------------------------
# general params
# ---------------------------
list_algo = ["DQN", "DQN-B", "PPO-B", "DDPG", "A2C-B", "GOFAI"]  # a new algo needs to be added to this list for backup to back up its results
nb_max_episodes_steps = 600  # pay attention, this could be changed to a constant divided by the action rate if its keeps increasing.
#This way we could use a fixed time insatead of a fixed amount of actions
# assert(nb_max_episodes_steps > 16 )
success_distance_to_goal = 2
slow_down_activation_distance = 2.5 * success_distance_to_goal  # detrmines at which distant we will punish the higher velocities

# ---------------------------
# training params
# ---------------------------
runs_to_do = 1
i_run = 1#this needs to be the same value as runs_to_do
assert(runs_to_do == i_run)
buffer_size = 50000  #replay buffer: this affects critically the iteration speed as the buffer gets filled (for dqn airsim)
use_checkpoint = True
training_steps_cap = 500000
nb_steps_warmup = 5000 #iterations are really fast during this phase
curriculum_learning = True
verbose = True

# ---------------------------
# testing params
# ---------------------------
testing_nb_episodes_per_model = max_zone*update_zone_window  # note that if number of zones are x, #pay attention
random.seed(hard_range_dic["Seed"][0])
deterministic = False
goals_list = []
for i in range(testing_nb_episodes_per_model+1):
    x_goal = random.choice(range(-49,49))
    y_goal = random.choice(range(-49,49))
    goals_list.append([x_goal, y_goal, 0])
goals_idx = 0
# then model get tested testing_nb_episodes_per_model/x
# times per zone
testing_nb_episodes_per_zone = int(testing_nb_episodes_per_model / max_zone)
assert(testing_nb_episodes_per_zone <= testing_nb_episodes_per_model), "get the equality right ,darn it"

# ---------------------------
# plotting params
# ---------------------------
average_runs = True
visualize_actions = False
# ---------------------------
# reseting params
# ---------------------------
connection_count_threshold = 20  # the upper bound to try to connect to multirouter
restart_game_from_scratch_count_threshold = 3  # the upper bound to try to reload unreal from scratch
window_restart_ctr_threshold = 2  # how many times we are allowed to restart the window
# before easying up the randomization

ease_constant = 0  # used when not meeting a zone for window_restart_ctr_threshold times. scales the randomization freq


# ---------------------------
# meta data  reload for reproducability
# ---------------------------
use_preloaded_json = False
meta_data_folder = "C:\workspace\zone2"

#--------------------------------
# profiling (of the env.step fnc)
#--------------------------------

profile = True



#--------------------------------
# Unreal game settings
#--------------------------------
game_resX = 640
game_resY = 480
ue4_winX = 640
ue4_winY = 480

#--------------------------------
# Multi-Modal Input settings
#--------------------------------
concatenate_inputs = True
encoded_depth = True
goal_position = True
velocity = True
grey = False
rgb = False

encoded_depth_H = 154
encoded_depth_W = 256
position_depth = 3
rgb_H = 144
rgb_W = 256
rgb_C = 3
grey_H = 144
grey_W = 256