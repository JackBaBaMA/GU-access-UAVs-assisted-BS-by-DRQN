import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import random
import scipy.integrate as integrate
import scipy.special as special
import torch.nn.functional as F
from collections import deque


select_mode = 'DQN'
#select_mode = 'QL-epsilon = 0.1'
#select_mode = 'Random'
#select_mode = 'QL-epislon = 0.3'

alpha_q = 0.8
gamma_q = 0.2

num_episodes = 10000



grid_num = 50
network_area = (800,800)
input_dim = len(network_area)   # state/input dim
num_State = (network_area[0] / grid_num ) ** 2



num_BSs = 2
num_UAVs = 2
BS_limitation = 3000
UAV_limitation = 300
BS_coverage_radius = 400
UAV_coverage_radius = 100
action_space = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 1), 
                (2, 2), (3, 3), (4, 4), (1, 0), (2, 0), 
                (3, 0), (4, 0), (1, -1), (2, -2), (3, -3), 
                (4, -4), (0, -1), (0, -2), (0, -3), (0, -4), 
                (-1, -1), (-2, -2), (-3, -3), (-4, -4), (-1, 0), 
                (-2, 0), (-3, 0), (-4, 0), (-1, 1), (-2, 2), 
                (-3, 3), (-4, 4), (0, 0)]

num_Actions = len(action_space)  
output_dim = num_Actions  # action/output dim 

moving_step = grid_num

# learning Hyperparameters
learning_rate = 0.005

gamma = 0.95
epsilon_start = 1.0
epsilon_end = 0.06
epsilon_decay = 0.999
target_update_freq = 100
batch_size = 32
buffer_capacity = 2000



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128) # create fully connected layers
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))   # ReLU (Rectified Linear Unit): ReLU is an activation function defined as f(x) = max(0, x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


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
        self.accessed_GUs_old = [0]
        self.state = []
        self.action = []
        self.reward = []
        self.recent_rewards = deque(maxlen=100)    # a deque for storing recent rewards
        self.next_position = []
        self.next_state = []
        self.tot_reward = 0
        self.max_min_reward = [0,0]   # for cliping rewards 
        self.Q_table = {}


#num_GUs = 1500
accessed_GUs_all_epi = []
num_GUsl = [8000,10000]
#num_GUsl = [(i + 1) * 500 for i in range(12) ]
#num_UAVsl = [2,3,4,5,6]
#num_BSsl = [2,3,4,5,6]
#num_BSsl = [7,8,9,10,11,12,13,14,15]

#num_UAVs = 2


for num_GUs in num_GUsl:
    accessed_GUs_1epi = 0 # for collecting accessed GUs number

    

    UAVs = [UAV() for _ in range(num_UAVs)]
    BSs = [BS() for _ in range(num_BSs)]
    GUs = [GU() for _ in range(num_GUs)]


    def Relocated_GUs():

        # Coordinates of density peaks, choose one or two peaks first
        #peak1 = (x / 4, y / 4)
        #peak2 = (3 * x / 4, 3 * y / 4)
        peak2 = peak1 = (network_area[0] / 2, network_area[0] / 2)

        # Create empty lists to store node positions
        node_positions_x = []
        node_positions_y = []

        # Define peak standard deviations
        peak_std_deviations = range(4,28,2)
        num_layer = len(peak_std_deviations)

        for std in peak_std_deviations:
            for _ in range(int(num_GUs/num_layer)):
                if np.random.rand() < 0.8:
                    # Randomly choose between the two peaks
                    if np.random.rand() < 0.5:
                        node_x = np.random.normal(peak1[0], std)
                        node_y = np.random.normal(peak1[1], std)
                    else:
                        node_x = np.random.normal(peak2[0], std)
                        node_y = np.random.normal(peak2[1], std)
                else:
                    # Randomly distribute nodes across the entire area
                    node_x = np.random.uniform(0, network_area[0])
                    node_y = np.random.uniform(0, network_area[0])

                node_positions_x.append(node_x)
                node_positions_y.append(node_y)

        node_positions_x = [0 if node < 0 else 800 if node > 800 else node for node in node_positions_x]
        node_positions_y = [0 if node < 0 else 800 if node > 800 else node for node in node_positions_y]

        for gu, nodex, nodey in zip(GUs, node_positions_x, node_positions_y):
            gu.position[0] = nodex
            gu.position[1] = nodey

        # Create a scatter plot to visualize the node distribution


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
                    if outage_prob < 0.8 and len(bs.accessed_GUs) < BS_limitation:
                        gu.accessed = True
                        bs.accessed_GUs.append(gu)
                        #print('gus in bs', len(bs.accessed_GUs) )     

            if not gu.accessed:
                for uav in UAVs:
                    distance_GU_UAV = np.sqrt(np.sum((gu.position - uav.position)**2))
                    if distance_GU_UAV <= UAV_coverage_radius:
                        outage_prob = calculate_GU2UAV_outage(distance_GU_UAV)
                        #print("UAV outage prob:", outage_prob) 
                        if outage_prob < 0.95 and len(uav.accessed_GUs) < UAV_limitation:
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


    # Initialize replay bufferqq
    UAVs_Reply = [ReplayBuffer(buffer_capacity) for _ in range(num_UAVs)]

    # Exploration-exploitation strategy (epsilon-greedy)
    epsilon = epsilon_start
    def select_action(uav_to_uav_learning, state):
        global epsilon
        if random.random() < epsilon:
            return random.choice(range(len(action_space)))    # random action
        else:
            with torch.no_grad():
                q_values = uav_to_uav_learning(torch.FloatTensor(state))
                return q_values.argmax().item()     # or the one with largest q-value

    def take_action(position, action):
        action_real = action_space[action]
        next_position = position + np.array(action_real) * moving_step # take action to fly 
        
        #if max(next_position) > network_area[0] or min(next_position) < 0:
        #    next_position = position

        #print(next_position)
            
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



    for episode in range(num_episodes):

        for uav in UAVs:    
            uav.tot_reward = 0


        if select_mode == 'DQN':
            # Create a dictionary to associate UAV objects with UAV_Learning objects
            uav_to_uav_learning = {}
            for uav, uav_learning in zip(UAVs, UAVs_Learning):
                uav_to_uav_learning[uav] = uav_learning   # store learning model for selecting action

            for uav in UAVs:
                
                uav.state = uav.position // grid_num
                #print(uav.state)

                uav.action = select_action(uav_to_uav_learning[uav], uav.state)  

                uav.next_state, uav.next_position, uav.accessed_GUs = take_action (uav.position, uav.action) # update accessing operation is in this step
                
                if len(uav.accessed_GUs) > UAV_limitation * 0:
                    uav.reward = (len(uav.accessed_GUs)) * ((network_area[0] / grid_num) / UAV_limitation) 
                else:
                    uav.reward = (len(uav.accessed_GUs) - len(uav.accessed_GUs_old)) * ((network_area[0] / grid_num) / UAV_limitation)  # reward is the difference value between length of two steps of accessed_GUs
                
                if max(uav.next_position) > 1200 or min(uav.next_position) < -400:    # relocate UAVs if out of boundary
                    uav.next_position = np.random.randint(0, network_area[0], size = (2,))
                    uav.next_state = uav.next_position // grid_num
                    uav.reward = - (len(uav.accessed_GUs)) * ((network_area[0] / grid_num) / UAV_limitation) 

                

                uav.accessed_GUs_old = uav.accessed_GUs
                #print(len(uav.accessed_GUs))
                accessed_GUs_1epi += len(uav.accessed_GUs)  
                #uav.tot_reward += uav.reward


            for bs in BSs:
                accessed_GUs_1epi += len(bs.accessed_GUs) 

            print(accessed_GUs_1epi)
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
                uav.tot_reward += uav.reward            
                uav.position = uav.next_position
                uav.state = uav.next_state


        if select_mode == 'QL-epislon = 0.3':
            
            for uav in UAVs:
                epsilon_dreedy = 0.3
                uav.state = uav.position // grid_num
                #print(uav.state)

                # initialize q table if state is new
                state_key = tuple(uav.state)   #check if state is in q-table already            
                if state_key not in uav.Q_table:
                    uav.Q_table[state_key] = [random.random()/10 for _ in range(num_Actions)]

                #Select action
                if random.random() < epsilon_dreedy:
                    uav.action = random.choice(range(num_Actions)) 
                else:
                    uav.action = np.argmax(uav.Q_table[state_key])

                moving_step_adjust = moving_step
                uav.next_state, uav.next_position, uav.accessed_GUs = take_action (uav.position, uav.action) # update accessing operation is in this step

                next_state_key = tuple(uav.next_state)  # check if next_state is in Q-table already
                if next_state_key not in uav.Q_table:
                    uav.Q_table[next_state_key] = [random.random()/10 for _ in range(num_Actions)] 


                if len(uav.accessed_GUs) > UAV_limitation * 0:
                    uav.reward = (len(uav.accessed_GUs)) * ((network_area[0] / grid_num) / UAV_limitation) 
                else:
                    uav.reward = (len(uav.accessed_GUs) - len(uav.accessed_GUs_old)) * ((network_area[0] / grid_num) / UAV_limitation)  # reward is the difference value between length of two steps of accessed_GUs     
                
                if max(uav.next_position) > 1200 or min(uav.next_position) < -400:    # relocate UAVs if out of boundary
                    uav.next_position = np.random.randint(0, network_area[0], size = (2,))
                    uav.next_state = uav.next_position // grid_num
                    uav.reward = - (len(uav.accessed_GUs)) * ((network_area[0] / grid_num) / UAV_limitation) 

                
                accessed_GUs_1epi += len(uav.accessed_GUs) 


                uav.accessed_GUs_old = uav.accessed_GUs

                #Update Q-table for Q(s,a)
                uav.Q_table[state_key][uav.action] = (1 - alpha_q) * uav.Q_table[state_key][uav.action] + alpha_q * (uav.reward + gamma_q * max(uav.Q_table[next_state_key]))
                
                uav.tot_reward += uav.reward 
                uav.position = uav.next_position
                uav.state = uav.next_state
                #print(uav.Q_table)

            for bs in BSs:
                accessed_GUs_1epi += len(bs.accessed_GUs) 

        if select_mode == 'QL-epsilon = 0.1':
            
            for uav in UAVs:

                uav.state = uav.position // grid_num
                #print(uav.state)

                # initialize q table if state is new
                state_key = tuple(uav.state)   #check if state is in q-table already            
                if state_key not in uav.Q_table:
                    uav.Q_table[state_key] = [random.random()/10 for _ in range(num_Actions)]

                #Select action
                if random.random() < epsilon:
                    uav.action = random.choice(range(num_Actions)) 
                else:
                    uav.action = np.argmax(uav.Q_table[state_key])

                
                moving_step_adjust = moving_step
                uav.next_state, uav.next_position, uav.accessed_GUs = take_action (uav.position, uav.action) # update accessing operation is in this step

                next_state_key = tuple(uav.next_state)  # check if next_state is in Q-table already
                if next_state_key not in uav.Q_table:
                    uav.Q_table[next_state_key] = [random.random()/10 for _ in range(num_Actions)] 

                if len(uav.accessed_GUs) > UAV_limitation * 0:   # ajust reward if training
                    uav.reward = (len(uav.accessed_GUs)) * ((network_area[0] / grid_num) / UAV_limitation) 
                else:
                    uav.reward = (len(uav.accessed_GUs) - len(uav.accessed_GUs_old)) * ((network_area[0] / grid_num) / UAV_limitation)  # reward is the difference value between length of two steps of accessed_GUs
            
                if max(uav.next_position) > 1200 or min(uav.next_position) < -400:    # relocate UAVs if out of boundary
                    uav.next_position = np.random.randint(0, network_area[0], size = (2,))
                    uav.next_state = uav.next_position // grid_num
                    uav.reward = - (len(uav.accessed_GUs)) * ((network_area[0] / grid_num) / UAV_limitation) 

                uav.accessed_GUs_old = uav.accessed_GUs
                accessed_GUs_1epi += len(uav.accessed_GUs)

                #Update Q-table for Q(s,a)
                uav.Q_table[state_key][uav.action] = (1 - alpha_q) * uav.Q_table[state_key][uav.action] + alpha_q * (uav.reward + gamma_q * max(uav.Q_table[next_state_key]))
                
                uav.tot_reward += uav.reward 
                uav.position = uav.next_position
                uav.state = uav.next_state
                #print(uav.Q_table)

            for bs in BSs:
                accessed_GUs_1epi += len(bs.accessed_GUs)
                

        if select_mode == 'Random':
            for uav in UAVs:                    

                uav.state = uav.position // grid_num
                #print(uav.state)
            
                #Select action
                uav.action = random.choice(range(len(action_space)))
                moving_step_adjust = moving_step

                uav.next_state, uav.next_position, uav.accessed_GUs = take_action (uav.position, uav.action) # update accessing operation is in this step
                uav.reward = (len(uav.accessed_GUs) - len(uav.accessed_GUs_old)) * ((network_area[0] / grid_num) / UAV_limitation)  # reward is the difference value between length of two steps of accessed_GUs
                if max(uav.next_position) > 1200 or min(uav.next_position) < -400:    # relocate UAVs if out of boundary
                    uav.next_position = np.random.randint(0, network_area[0], size = (2,))
                    uav.next_state = uav.next_position // grid_num
                    uav.reward = - (len(uav.accessed_GUs)) * ((network_area[0] / grid_num) / UAV_limitation) 
                uav.accessed_GUs_old = uav.accessed_GUs            
        
                uav.position = uav.next_position
                uav.state = uav.next_state
                uav.tot_reward += uav.reward 

                accessed_GUs_1epi += len(uav.accessed_GUs)
            for bs in BSs:
                accessed_GUs_1epi += len(bs.accessed_GUs)


        # Epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

            

        for uav in UAVs:

            # try:
            #     with open('output.txt', "a") as file:  
            #       #file.write(f"Episode {episode + 1}: {uav.tot_reward}\n")
            #       file.write(f"{uav.tot_reward}\n")
            # finally:
            #     file.close()
            print(f"Episode {episode + 1} - Total Reward: {uav.tot_reward}")

    print(accessed_GUs_1epi)
    accessed_GUs_all_epi.append(accessed_GUs_1epi)

print(accessed_GUs_all_epi)

try:
    with open('output.txt', "a") as file:
        file.write(str(accessed_GUs_all_epi))
finally:
    file.close()




# Save the trained DQN model
#for uav in UAVs_Learning:
#    torch.save(uav.state_dict(), 'dqn_model.pth')
   


