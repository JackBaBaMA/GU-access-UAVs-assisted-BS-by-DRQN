import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import math
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
import random
import scipy.integrate as integrate
import scipy.special as special
from get_real_action import get_real_action

# Define the parameters
num_GUs = 1500
num_UAVs = 3
num_BSs = 2
BS_limitation = 500
UAV_limitation = 300
BS_coverage_radius = 400
UAV_coverage_radius = 100
moving_step = 100
network_area = (800, 800)
#max_num_steps = 1000
gamma = 0.9  # Discount factor
num_actions_per_UAV = 4  # Four flying directions: north, south, west, east
num_actions_coordinated = num_actions_per_UAV * num_UAVs # one big action for all the UAVS 


#parameters for bs2gu
# Define the GU class
class GU:
    def __init__(self):
        
        self.position = np.random.randint(0, network_area[0], size=(2,))  # Random position (x, y)
        self.accessed = False

# Define the BS class
class BS:
    def __init__(self):
        self.position = np.random.randint(0, network_area[0], size=(2,))  # Random position in the edge
        bs_axis_elimination  = random.randint(0,1)

        if bs_axis_elimination == 0:
            self.position[0] = 0
        else:
            self.position[1] = 0

        self.accessed_GUs = []

# Define the UAV class

def get_action_per_UAV(action_implicit):
        max_index = np.argmax(action_implicit)
        newer_array = np.zeros_like(action_implicit)
        newer_array[0, max_index] = 1
        return newer_array

class UAV:
    def __init__(self, UAV_id):
        self.id = UAV_id

        self.position = np.random.randint(0, network_area[0], size=(2,))  # Random position (x, y)

        self.accessed_GUs = []
        action_implicit = np.random.randint(1, 100 * num_actions_per_UAV, size = (1, num_actions_per_UAV))
        self.action = get_action_per_UAV(action_implicit)        
  
    def get_state_representation(self):
        #return [self.id, self.position[0], self.position[1], len(self.accessed_GUs)]
        grid_num_state = moving_step # grid 10 to reduce state number 
        return [self.position[0] // grid_num_state, self.position[1] // grid_num_state]




# Create GU, BS, and UAV instances
GUs = [GU() for _ in range(num_GUs)]
BSs = [BS() for _ in range(num_BSs)]
UAVs = [UAV(UAV_id) for UAV_id in range(num_UAVs)]

# relocate positions of GUs to build two peaks

area_size = network_area[0]

# Number of total nodes
total_nodes_ini = num_GUs / 2
total_nodes = int(total_nodes_ini / 2)

# Create a random distribution of nodes
nodes_x = np.random.rand(total_nodes) * area_size
nodes_y = np.random.rand(total_nodes) * area_size

# Create two densely populated peaks
peak1_x = np.random.normal(area_size/4, 5, total_nodes//2)
peak1_y = np.random.normal(area_size/4, 5, total_nodes//2)

peak2_x = np.random.normal(3*area_size/4, 5, total_nodes//2)
peak2_y = np.random.normal(3*area_size/4, 5, total_nodes//2)

# Combine the random nodes with the two peaks
nodes_x = np.concatenate((nodes_x, peak1_x, peak2_x))
nodes_y = np.concatenate((nodes_y, peak1_y, peak2_y))
        
for gu in GUs:
    gu.position[0] = nodes_x[0]
    gu.position[0] = nodes_y[1]

""" plt.figure(figsize=(8, 8))
plt.scatter(nodes_x, nodes_y, s=10, alpha=0.5)
plt.title("Random Node Distribution with Peaks")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.show() """






# Function to calculate Outage Probability (OP) between GU and BS
def calculate_GU2BS_outage(d):
    P_t = 1  # Transmit power in Watts
    G_t = 5.0  # Transmitter antenna gain
    G_r = 4.0  # Receiver antenna gain
    beta = 0.5  # Channel gain
    PL_0 = 1.0  # Path loss at the reference distance of 1 meter in dB
    eta = 1.5  # Path loss exponent
    shadow_std = 2  # Log-normal shadowing standard deviation in dB
    N_0 = 1e-9  # Noise power spectral density in Watts/Hz
    W = 1e6  # Bandwidth in Hz
    I_tot = 1e-6  # Total interference power in Watts
    gamma_th = 5.5  # Threshold SINR for outage
    def gaussian_q(x):
        return 1-1 * special.erfc(x / np.sqrt(2))

    def path_loss(d, PL_0, eta, shadow_std):
        return PL_0 + 10 * eta * np.log10(d) + np.random.normal(0, shadow_std)

    def sinr(P_t, G_t, G_r, beta, d, PL_0, eta, shadow_std, N_0, W, I_tot):
        PL_d = path_loss(d, PL_0, eta, shadow_std)
        SINR = (P_t * G_t * G_r * beta) / (10**(PL_d/10) * N_0 * W + I_tot)
        return SINR

    def outage_probability(P_t, G_t, G_r, beta, d, PL_0, eta, shadow_std, N_0, W, I_tot, gamma_th):
        SINR = sinr(P_t, G_t, G_r, beta, d, PL_0, eta, shadow_std, N_0, W, I_tot)
        return gaussian_q(np.sqrt(0.5 * gamma_th / SINR))
    
    result = outage_probability(P_t, G_t, G_r, beta, d, PL_0, eta, shadow_std, N_0, W, I_tot, gamma_th)

    return result

# Function to calculate Outage Probability (OP) between GU and UAV
def calculate_GU2UAV_outage(distance_GU_UAV):
    P_t = 0.1  # Transmit power
    G_UAV = 1.0  # Antenna gain of UAV
    G_GU = 1.0  # Antenna gain of ground user
    N_0 = 1e-9  # Noise power spectral density
    d_0 = 1.0  # Reference distance
    m = 2  # Nakagami-m fading parameter
    X_shadow = 1.0  # Log-normal shadowing component in dB
    gamma_th = 5.5  # Threshold SINR for outage
    d = distance_GU_UAV


    def q_function(x):
        return 0.5 * special.erfc(x / np.sqrt(2))

    def pdf_sinr(gamma, m):
        numerator = (m**m) * (P_t * G_UAV * G_GU / (N_0 * d_0**m))**m
        denominator = special.gamma(m) * (m * gamma + 1)**(m + 1)
        return numerator / denominator * 10**(-X_shadow / 10)
    
    def integrand(gamma):
        return q_function(np.sqrt(2 * gamma_th * (N_0 * 50 * d**m / (P_t * G_UAV * G_GU) + 1))) * pdf_sinr(gamma, m)
    result = integrate.quad(integrand, 0, np.inf)[0]
    result = 1 - result/3618626508500.2524
    return result

# Function to update state after UAV movement
def update_state():
    # Check which GUs are within the coverage radius of UAVs and BSs
    for gu in GUs:

        gu.accessed = False

        for bs in BSs:
            distance_GU_BS = math.sqrt((gu.position[0] - bs.position[0]) ** 2 + (gu.position[1] - bs.position[1]) ** 2)
            if distance_GU_BS <= BS_coverage_radius:
                outage_prob = calculate_GU2BS_outage(distance_GU_BS)     
                #print("bs outage prob:", outage_prob)           
                if outage_prob < 0.8 and len(bs.accessed_GUs) < BS_limitation:
                    gu.accessed = True
                    bs.accessed_GUs.append(gu)
                    #print('gus in bs', len(bs.accessed_GUs) )
        
        

        if not gu.accessed:
            for uav in UAVs:
                distance_GU_UAV = math.sqrt((gu.position[0] - uav.position[0]) ** 2 + (gu.position[1] - uav.position[1]) ** 2)
                if distance_GU_UAV <= UAV_coverage_radius:
                    outage_prob = calculate_GU2UAV_outage(distance_GU_UAV)
                    #print("UAV outage prob:", outage_prob) 
                    if outage_prob < 0.3 and len(uav.accessed_GUs) < UAV_limitation:
                        gu.accessed = True
                        uav.accessed_GUs.append(gu)

def get_state():
    result = [UAV.get_state_representation() for UAV in UAVs]
    return result
        

#initialize state

update_state()
state = get_state()

""" input_for_drqn = prepare_input(UAVs) """

# Create the DRQN model
timesteps = num_UAVs 
data_dim = 2     # state len for one UAV
input_shape = (timesteps, data_dim)
def create_drqn_model(input_shape, num_actions):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))   # 64 LSTM cells or neurons
    model.add(LSTM(64))  # Add another LSTM layer
    model.add(Dense(100, activation='relu'))   # 100 represents the number of neurons 
    model.add(Dense(100, activation='relu'))     # Add a dense hidden layer
    model.add(Dense(100, activation='relu')) 
    model.add(Dense(num_actions, activation='linear'))
    return model
drqn_model = create_drqn_model(input_shape, num_actions_coordinated)

# Define the loss function and optimizer
loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

# Define the function to update the DRQN model
def update_drqn_model(state, action, new_state, reward):
    state = tf.convert_to_tensor(state, dtype=tf.float32)
    state = tf.reshape(state, (-1, timesteps, data_dim))
    new_state = tf.convert_to_tensor(new_state, dtype=tf.float32)
    new_state = tf.reshape(new_state, (-1, timesteps, data_dim))
    reduce_max_target = tf.reduce_max(drqn_model.predict(new_state))

    #print (state)
    #print (new_state)
    
    #print (reduce_max_target)
    target = reward + gamma * reduce_max_target
    with tf.GradientTape() as tape:
        predicted = tf.reduce_sum(drqn_model(state) * tf.one_hot(action, num_actions_coordinated), axis=1)
        
        #print(target)
        #print(predicted)
        loss = loss_function(tf.convert_to_tensor([target]), predicted) # make target and predicted have the same shape
    gradients = tape.gradient(loss, drqn_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, drqn_model.trainable_variables))

# Define the function to make a decision based on the current state
def make_decision(state):
    state = tf.convert_to_tensor(state, dtype=tf.float32)
    state = tf.reshape(state, (-1, timesteps, data_dim))
    action = tf.argmax(drqn_model.predict(state)[0])
    return action.numpy()

# Define the function to take an action and observe the new state and reward

def take_action(state, coordinated_action):
    #update actions by the output coordinated action

    real_action_coordinated = get_real_action(coordinated_action)

    real_action_coordinated_reshaped = real_action_coordinated.reshape(num_UAVs, num_actions_per_UAV)

    for uav_idx, uav in enumerate(UAVs):
        action_UAV = real_action_coordinated_reshaped[uav_idx]
        uav.action = action_UAV
        moving_direction = np.argmax(uav.action)            
        

        # Update the UAV positions
        if moving_direction == 0:  # North
            uav.position[1] += moving_step
            if uav.position[1] > network_area[0]: # in case UAVs escaping from the network area
                uav.position[1] -= moving_step
        elif moving_direction == 1:  # South
            uav.position[1] -= moving_step
            if uav.position[1] < 0:
                uav.position[1] += moving_step
        elif moving_direction == 2:  # West
            uav.position[0] -= moving_step
            if uav.position[0] < 0:
                uav.position[0] += moving_step
        elif moving_direction == 3:  # East
            uav.position[0] += moving_step
            if uav.position[0] > network_area[0]:
                uav.position[0] -= moving_step

    # Update  the accessed GUs

    update_state()
    new_state = get_state()
    reward = sum(len(UAV.accessed_GUs) for UAV in UAVs)
    print(reward)

    for uav_idx, uav in enumerate(UAVs):
        uav.accessed_GUs = []

    for gu in GUs:
        gu.accessed = False

    for bs in BSs:
        bs.accessed_GUs = []

    return new_state, reward


# Run the simulation for a number of episodes
num_episodes = 100
max_num_steps = 100
tot_reward = []
reward_old = 0
for episode in range(num_episodes):

    state = get_state()
    print(state)
    state = tf.convert_to_tensor(state, dtype=tf.float32)
    state = tf.reshape(state, (-1, timesteps, data_dim))

    total_reward = 0

    for step in range(max_num_steps):
        action = make_decision(state)
        new_state, reward_new = take_action(state, action)
        reward = reward_new - reward_old    # set the reward as the difference of accessed GUs numbers 
        reward_old = reward_new    
        total_reward += reward
        update_drqn_model(state, action, new_state, reward)
        state = new_state

    print(f"Episode {episode+1}, Total Reward: {total_reward}")
    tot_reward.append(total_reward)
    file_name = "tot_reward.txt"
    try:
        with open(file_name, "a") as file:  # "a" mode for appending to the file
            # Step 2: Write the value to the file
            file.write(f"Episode {episode + 1}: {tot_reward}\n")
    finally:
        # Step 3: Close the file (even if an exception occurs)
        file.close()
    

    # Plot the UAVs positions after the episode
"""     plt.figure()
    plt.title(f"UAV positions after Episode {episode+1}")
    for uav_idx, uav in enumerate(UAVs):
        plt.scatter(uav.position[0], uav.position[1], label=f"UAV {uav_idx}")
    for bs in BSs:
        plt.scatter(bs.position[0], bs.position[1], label="BS", marker='s', s=100)
    for gu in GUs:
        plt.scatter(gu.position[0], gu.position[1], label="GU", marker='x', s=50)
    plt.legend()
    plt.xlim(0, network_area[0])
    plt.ylim(0, network_area[1])
    plt.show()
 """