directions = [
    (0, 1),    # North (y + 1)
    (1, 1),    # Northeast (x + 1, y + 1)
    (1, 0),    # East (x + 1)
    (1, -1),   # Southeast (x + 1, y - 1)
    (0, -1),   # South (y - 1)
    (-1, -1),  # Southwest (x - 1, y - 1)
    (-1, 0),   # West (x - 1)
    (-1, 1),    # Northwest (x - 1, y + 1)
    (0, 0)
]

movements = []

# Generate movements for each direction and magnitude
for direction in directions:
    x, y = direction
    for magnitude in range(1, 5):  # Magnitude from 1 to 8
        movements.append((x * magnitude, y * magnitude))

# Print the generated movements
print(movements)
