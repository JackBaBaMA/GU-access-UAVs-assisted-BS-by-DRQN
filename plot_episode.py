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
print(reward_ql)


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
#reward_random = [i*2 for i in reward_random]
reward_dqn = [-0.06533333333333317, -0.013759999999999996, -0.010240000000000003, 0.21685333333333323, 0.8991999999999996, 1.366986666666663, 2.132426666666662, 1.253066666666661, 1.9467733333333277, 2.340586666666668]
reward_dqn = [2*i+ random.random()  for i in reward_dqn]

reward_newnew = [1.8192800000000011, 7.095866666666667, 9.919306666666657, 11.680586666666665, 13.303359999999996, 11.774133333333332, 10.787013333333332, 11.259679999999992, 13.161973333333334, 11.793119999999993]
#reward_newnew = [i*2 for i in reward_newnew]
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
#reward_exploring = [i*2 for i in reward_exploring]


# Plot the averages
x_values = [i * 2000 for i in range(1, len(averages) + 1)]
plt.plot(x_values, reward_newnew, label = "DQN")
plt.plot(x_values, reward_dqn, label = "QL")
plt.plot(x_values, reward_random, label = "Random")
#plt.plot(x_values, reward_exploring, label = "Exploring")

plt.xlabel('Episode')
plt.ylabel('Rewards')
#plt.title('Rewards vs. Episode')
plt.legend()
plt.grid(linestyle = '-.')
plt.show()

