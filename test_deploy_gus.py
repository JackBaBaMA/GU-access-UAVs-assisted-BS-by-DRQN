import numpy as np
import matplotlib.pyplot as plt

# Define the size of the area
x, y = 100, 100

# Number of nodes to deploy
n = 1000


def relocated_GUs():
    # Coordinates of density peaks, choose one or two peaks first
    #peak1 = (x / 4, y / 4)
    #peak2 = (3 * x / 4, 3 * y / 4)
    peak2 = peak1 = (x / 2, y / 2)

    # Create empty lists to store node positions
    node_positions_x = []
    node_positions_y = []

    # Define peak standard deviations
    peak_std_deviations = range(4,28,2)
    num_layer = len(peak_std_deviations)

    for std in peak_std_deviations:
        for _ in range(int(n/num_layer)):
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
                node_x = np.random.uniform(0, x)
                node_y = np.random.uniform(0, y)

            node_positions_x.append(node_x)
            node_positions_y.append(node_y)

    node_positions_x = [0 if node < 0 else 800 if node > 800 else node for node in node_positions_x]
    node_positions_y = [0 if node < 0 else 800 if node > 800 else node for node in node_positions_y]

    return node_positions_x, node_positions_y

node_positions_x, node_positions_y = relocated_GUs()



# Create a scatter plot to visualize the node distribution
plt.figure(figsize=(8, 6))
plt.scatter(node_positions_x, node_positions_y, s=5, alpha=0.7)
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Node Distribution with Varying Density (More Nodes Near Peaks)')
plt.show()
