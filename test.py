import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define a simple neural network model for DQN
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

# Define a replay buffer to store and sample experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Initialize the environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update_freq = 10
batch_size = 64
buffer_capacity = 10000

# Initialize DQN networks
dqn = DQN(state_dim, action_dim)
target_dqn = DQN(state_dim, action_dim)
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()

# Initialize optimizer and loss function
optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Initialize replay buffer
replay_buffer = ReplayBuffer(buffer_capacity)

# Exploration-exploitation strategy (epsilon-greedy)
epsilon = epsilon_start
def select_action(state):
    global epsilon
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            q_values = dqn(torch.FloatTensor(state))
            return q_values.argmax().item()

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = select_action(state)
        result = env.step(action)
        next_state, reward, done, info = result[:4] 
        replay_buffer.push(state, action, reward, next_state, done)

        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)

            q_values = dqn(states)
            next_q_values = target_dqn(next_states)
            next_max_q_values = next_q_values.max(1)[0]

            target_q_values = q_values.clone()
            target_q_values[np.arange(batch_size), actions] = rewards + gamma * next_max_q_values * ~dones

            loss = loss_fn(q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if episode % target_update_freq == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        total_reward += reward
        state = next_state

        if done:
            break

    # Epsilon decay
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    print(f"Episode {episode + 1} - Total Reward: {total_reward}")

# Save the trained DQN model
torch.save(dqn.state_dict(), 'dqn_model.pth')
