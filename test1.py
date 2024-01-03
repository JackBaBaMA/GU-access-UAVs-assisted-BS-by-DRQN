import matplotlib.pyplot as plt

# Original curve represented as a list of points (x, y)
original_curve = [(1, 1), (2, 2), (3, 4), (4, 8), (5, 16)]

# Scaling factor to make the curve bigger
scaling_factor = 2

# Apply the scaling transformation to each point
scaled_curve = [(x * scaling_factor, y * scaling_factor) for x, y in original_curve]

# Extract the x and y coordinates of the scaled curve
scaled_x, scaled_y = zip(*scaled_curve)

# Plot the original and scaled curves
plt.plot(*zip(*original_curve), label="Original Curve")
plt.plot(scaled_x, scaled_y, label="Scaled Curve")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
