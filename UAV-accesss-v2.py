import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import random
import scipy.integrate as integrate
import scipy.special as special
import torch.nn.functional as F



grid_num = 100
network_area = (800,800)
input_dim = 2   # state/input dim
num_Actions = 5  
output_dim = num_Actions  # action/output dim 

num_GUs = 2000
num_BSs = 2
num_UAVs = 2
BS_limitation = 500
UAV_limitation = 300
BS_coverage_radius = 400
UAV_coverage_radius = 100
action_space = ([0,0],[0,1],[0,-1],[1,0],[-1,0])
moving_step = grid_num

# learning Hyperparameters
learning_rate = 0.001
gamma = 0.8
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update_freq = 10
batch_size = 64
buffer_capacity = 10000

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):   # store them
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):   # take them
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return state, action, reward, next_state
    
    def __len__(self):
        return len(self.buffer)
    
    
class BS:
    def __init__(self):
        self.position = np.random.randint(0, network_area[0], size = (2,)) # random position in the edge of area
        bs_axis_elimination = random.randint(0,1)
        self.position[bs_axis_elimination] = 0  # eliminate one axis
        self.accessed_GUs = []


class GU:
    def __init__(self):
        self.position = np.random.randint(0, network_area[0], size = (2,))
        self.accessed = False


class UAV:
    def __init__(self):
        self.position = np.random.randint(0, network_area[0], size = (2,))
        self.accessed_GUs = []
        self.accessed_GUs_old = []
        self.state = []
        self.action = []
        self.reward = []
        self.next_position = []
        self.next_state = []
        self.tot_reward = 0
    def __str__(self):
        return f"GU at position ({self.position[0]}, {self.position[1]}), Accessed: {self.accessed_GUs}"



UAVs = [UAV() for _ in range(num_UAVs)]
BSs = [BS() for _ in range(num_BSs)]
GUs = [GU() for _ in range(num_GUs)]


def Relocated_GUs():
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
            
    for gu, nodex, nodey in zip(GUs, nodes_x, nodes_y):
        gu.position[0] = nodex
        gu.position[1] = nodey
    



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
    result = d / BS_coverage_radius
    #print(result)
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
    result = d / UAV_coverage_radius
    #print(result)
    return result

# Function to update state after UAV movement
def update_accessing():
    # Check which GUs are within the coverage radius of UAVs and BSs
    for gu in GUs:

        gu.accessed = False

        for bs in BSs:
            distance_GU_BS = np.sqrt(np.sum((gu.position - bs.position)**2))
            if distance_GU_BS <= BS_coverage_radius:
                outage_prob = calculate_GU2BS_outage(distance_GU_BS)     
                #print("bs outage prob:", outage_prob)           
                if outage_prob < 0.9 and len(bs.accessed_GUs) < BS_limitation:
                    gu.accessed = True
                    bs.accessed_GUs.append(gu)
                    #print('gus in bs', len(bs.accessed_GUs) )     

        if not gu.accessed:
            for uav in UAVs:
                distance_GU_UAV = np.sqrt(np.sum((gu.position - uav.position)**2))
                if distance_GU_UAV <= UAV_coverage_radius:
                    outage_prob = calculate_GU2UAV_outage(distance_GU_UAV)
                    #print("UAV outage prob:", outage_prob) 
                    if outage_prob < 0.7 and len(uav.accessed_GUs) < UAV_limitation:
                        gu.accessed = True
                        uav.accessed_GUs.append(gu)
                        


Relocated_GUs()
update_accessing()

# Initialize DQN networks
UAVs_Learning = [DQN(input_dim, output_dim) for _ in range(num_UAVs)]

UAVs_Target = [DQN(input_dim, output_dim) for _ in range(num_UAVs)]
for uav_target, uav_learning in zip(UAVs_Target, UAVs_Learning):
    uav_target.load_state_dict(uav_learning.state_dict())
    uav_target.eval()

# Initialize optimizer and loss function

UAVs_Optimizers = [torch.optim.Adam(uav.parameters(), lr=learning_rate) for uav in UAVs_Learning]
loss_fn = nn.MSELoss()


# Initialize replay buffer
UAVs_Reply = [ReplayBuffer(buffer_capacity) for _ in range(num_UAVs)]

# Exploration-exploitation strategy (epsilon-greedy)
epsilon = epsilon_start
def select_action(uav_to_uav_learning, state):
    global epsilon
    if random.random() < epsilon:
        return random.choice(range(4))    # random action
    else:
        with torch.no_grad():
            q_values = uav_to_uav_learning(torch.FloatTensor(state))
            return q_values.argmax().item()     # or the one with largest q-value

def take_action(position, action):
    action_real = action_space[action]
    next_position = position + np.array(action_real) * moving_step # take action to fly 
    if (next_position[0] > network_area[0]).any() or (next_position[0] < 0).any() or (next_position[1] > network_area[0]).any() or (next_position[1] < 0).any():
    # Handle out-of-bounds positions here
        next_position = position
    next_state = next_position // grid_num # to reduce state space by dividing grid_num
    for bs in BSs:                  # clear BSs and UAVs accessed_GUs 
        bs.accessed_GUs = []
    for uav in UAVs:
        uav.accessed_GUs = [] 
    update_accessing()     
    output_accessed_GUs = []
    access_GUs_dict = {(tuple(uav.position.tolist())): uav.accessed_GUs for uav in UAVs} # A dict for storage
    position_tuple = tuple(position) # convert numpy array into a tuple, bacause tuple is hashable
    if position_tuple in access_GUs_dict:
        output_accessed_GUs = access_GUs_dict[position_tuple]

    return next_state, next_position, output_accessed_GUs # the first output is next_state, sceond one is accessed_GUs, 

# Training loop
num_episodes = 10000

for episode in range(num_episodes):
    
    total_reward = 0

    # Create a dictionary to associate UAV objects with UAV_Learning objects
    uav_to_uav_learning = {}
    for uav, uav_learning in zip(UAVs, UAVs_Learning):
        uav_to_uav_learning[uav] = uav_learning         # store learning model for selecting action

    for uav in UAVs:
        uav.state = uav.position // grid_num
        print('current state', uav.state)
    
        uav.action = select_action(uav_to_uav_learning[uav], uav.state)
        uav.next_state, uav.next_position, uav.accessed_GUs = take_action (uav.position, uav.action) # update accessing operation is in this step
        
        uav.reward = len(uav.accessed_GUs) - len(uav.accessed_GUs_old)   # reward is the difference value between length of two steps of accessed_GUs
        #print('Imediate Reward', uav.reward)
        uav.accessed_GUs_old = uav.accessed_GUs
        #print('Current Accessed GUs', len(uav.accessed_GUs))
        uav.tot_reward += uav.reward
        
    # push buffers 
    # Assuming you have a list of UAV instances nameffd 'UAVs' and their corresponding replay buffers in 'ReplayBuffers'
    for uav, replay_buffer in zip(UAVs, UAVs_Reply):
        replay_buffer.push(uav.state, uav.action, uav.reward, uav.next_state)

    # the following is for training
    for uav_buffer, uav_learning, uav_target, uav_optimizer in zip(UAVs_Reply, UAVs_Learning, UAVs_Target, UAVs_Optimizers):
        if len(uav_buffer) > batch_size:
            states, actions, rewards, next_states = uav_buffer.sample(batch_size)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)

            q_values = uav_learning(states)
            
            next_q_values = uav_target(next_states)
            next_max_q_values = next_q_values.max(1)[0]

            target_q_values = q_values.clone()
            #print(actions)
            #print(target_q_values[np.arange(batch_size), actions])
            target_q_values[np.arange(batch_size), actions] = rewards + gamma * next_max_q_values

            loss = loss_fn(q_values, target_q_values)

            uav_optimizer.zero_grad()
            loss.backward()
            uav_optimizer.step()

            if episode % target_update_freq == 0:
                uav_target.load_state_dict(uav_learning.state_dict())

    for uav in UAVs:
        uav.position = uav.next_position
        uav.state = uav.next_state
    

    # Epsilon decay
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    for uav in UAVs:
        print(f"Episode {episode + 1} - Total Reward: {uav.tot_reward}")

# Save the trained DQN model
for uav in UAVs_Learning:
    torch.save(uav.state_dict(), 'dqn_model.pth')





    











    













