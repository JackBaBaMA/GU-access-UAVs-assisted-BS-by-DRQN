import matplotlib.pyplot as plt
import random
# Initialize variables
line_counter = 0
line_sum = 0
averages = []

# Open the file for reading
with open('output_dqn.txt', 'r') as file:
    for line in file:
        line_counter += 1
        value = float(line.strip())  # Assuming the data is numeric and each line contains a single value

        # Add the value to the sum
        line_sum += value

        # If you've processed 100 lines, calculate the average
        if line_counter == 2000:
            average = line_sum / line_counter
            averages.append(average)
            line_counter = 0
            line_sum = 0

# If there are remaining lines, calculate the final average
if line_counter > 0:
    average = line_sum / line_counter
    averages.append(average)

    
reward_dqn = averages


# the second one
line_counter = 0
line_sum = 0
averages = []
# Open the file for reading
with open('output_ql.txt', 'r') as file:
    for line in file:
        line_counter += 1
        value = float(line.strip())  # Assuming the data is numeric and each line contains a single value

        # Add the value to the sum
        line_sum += value

        # If you've processed 100 lines, calculate the average
        if line_counter == 2000:
            average = line_sum / line_counter
            averages.append(average)
            line_counter = 0
            line_sum = 0

# If there are remaining lines, calculate the final average
if line_counter > 0:
    average = line_sum / line_counter
    averages.append(average)

reward_ql = averages


# the third one 
line_counter = 0
line_sum = 0
averages = []
# Open the file for reading
with open('output_random.txt', 'r') as file:
    for line in file:
        line_counter += 1
        value = float(line.strip())  # Assuming the data is numeric and each line contains a single value

        # Add the value to the sum
        line_sum += value

        # If you've processed 100 lines, calculate the average
        if line_counter == 2000:
            average = line_sum / line_counter
            averages.append(average)
            line_counter = 0
            line_sum = 0

# If there are remaining lines, calculate the final average
if line_counter > 0:
    average = line_sum / line_counter
    averages.append(average)

reward_random = averages


reward_newnew = [1.8192800000000011, 7.095866666666667, 9.919306666666657, 11.680586666666665, 13.303359999999996, 15.774133333333332, 14.787013333333332, 15.259679999999992, 15.161973333333334, 15.793119999999993]

reward_exploring = [0.05333333333333334,
                    0.266,
                    0.32,
                    0.83,
                    5.62,
                    7.3,
                    12.13,
                    15.1,
                    16,
                    16]

# Plot the averages
x_values = [i * 2000 for i in range(1, len(averages) + 1)]
plt.plot(x_values, reward_newnew, label = "DQN")
plt.plot(x_values, reward_dqn, label = "QL")
plt.plot(x_values, reward_random, label = "Random")
plt.plot(x_values, reward_exploring, label = "Exploring")

plt.xlabel('Episode')
plt.ylabel('Reward')
#plt.title('Rewards vs. Episode')
plt.legend()
plt.show()

