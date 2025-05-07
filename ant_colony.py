import random
import numpy as np
import matplotlib.pyplot as plt

# Number of cities
n_cities = 5

# Randomly generate coordinates for the cities (can replace with real coordinates)
cities = np.random.rand(n_cities, 2)

# Distance function (Euclidean distance)
def euclidean_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

# Create the distance matrix
dist_matrix = np.array([[euclidean_distance(cities[i], cities[j]) for j in range(n_cities)] for i in range(n_cities)])

# ACO Parameters
alpha = 1  # Pheromone importance
beta = 2   # Distance priority
evaporation_rate = 0.5  # Pheromone evaporation rate
pheromone_deposit = 1.0  # Pheromone deposited by ants

# Initialize pheromone levels
pheromone = np.ones((n_cities, n_cities))

# ACO algorithm to solve TSP
def aco_algorithm(iterations, n_ants):
    best_path = None
    best_length = float('inf')
    
    for iteration in range(iterations):
        all_paths = []
        all_lengths = []

        # Each ant constructs a path
        for ant in range(n_ants):
            path = construct_path()
            length = calculate_path_length(path)
            all_paths.append(path)
            all_lengths.append(length)

            if length < best_length:
                best_path = path
                best_length = length

        update_pheromone(all_paths, all_lengths)
        pheromone * (1 - evaporation_rate)  # Evaporate pheromone

        print(f"Iteration {iteration + 1}, Best path length: {best_length}")

    return best_path, best_length

# Construct a path for an ant
def construct_path():
    path = [random.randint(0, n_cities - 1)]  # Start from a random city
    visited = set(path)
    
    while len(path) < n_cities:
        current_city = path[-1]
        probabilities = calculate_transition_probabilities(current_city, visited)
        next_city = random.choices(range(n_cities), weights=probabilities)[0]
        path.append(next_city)
        visited.add(next_city)

    path.append(path[0])  # Return to the starting city
    return path

# Calculate transition probabilities for an ant to move to the next city
def calculate_transition_probabilities(current_city, visited):
    probabilities = []
    total_pheromone = 0.0
    
    for next_city in range(n_cities):
        if next_city not in visited:
            pheromone_strength = pheromone[current_city][next_city] ** alpha
            distance_heuristic = (1.0 / dist_matrix[current_city][next_city]) ** beta
            total_pheromone += pheromone_strength * distance_heuristic

    for next_city in range(n_cities):
        if next_city not in visited:
            pheromone_strength = pheromone[current_city][next_city] ** alpha
            distance_heuristic = (1.0 / dist_matrix[current_city][next_city]) ** beta
            probability = (pheromone_strength * distance_heuristic) / total_pheromone
            probabilities.append(probability)
        else:
            probabilities.append(0)
    
    return probabilities

# Calculate the total length of a path
def calculate_path_length(path):
    length = 0.0
    for i in range(len(path) - 1):
        length += dist_matrix[path[i]][path[i + 1]]
    return length

# Update pheromone levels after each iteration
def update_pheromone(all_paths, all_lengths):
    global pheromone
    pheromone *= (1 - evaporation_rate)  # Evaporate pheromone

    for path, length in zip(all_paths, all_lengths):
        deposit = pheromone_deposit / length
        for i in range(len(path) - 1):
            pheromone[path[i]][path[i + 1]] += deposit
            pheromone[path[i + 1]][path[i]] += deposit  # Make it bidirectional

# Run the ACO algorithm
best_path, best_length = aco_algorithm(iterations=50, n_ants=10)

# Display the best path and its length
print("Best path:", best_path)
print("Best path length:", best_length)

# Plot the best path
x = [cities[i][0] for i in best_path]
y = [cities[i][1] for i in best_path]
plt.plot(x, y, marker='o')
plt.title(f"Best Path (Length: {best_length})")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()






"""
import random
import numpy as np
import matplotlib.pyplot as plt

# Number of cities
n_cities = 5

# Coordinates of the cities (can be replaced with actual city coordinates)
cities = np.random.rand(n_cities, 2)

# Distance matrix (Euclidean distance between cities)
def euclidean_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# Create the distance matrix
dist_matrix = np.array([[euclidean_distance(cities[i], cities[j]) for j in range(n_cities)] for i in range(n_cities)])

# Parameters for ACO
alpha = 1  # Pheromone importance
beta = 5   # Distance priority (the lower the value, the more the ants prefer shorter paths)
evaporation_rate = 0.5  # Evaporation rate (0-1)
pheromone_deposit = 1.0  # Amount of pheromone deposited by ants

# Initialize pheromone levels
pheromone = np.ones((n_cities, n_cities))  # Start with equal pheromone level on all paths

# Define the Ant Colony Optimization (ACO) algorithm
def aco_algorithm(iterations, n_ants):
    best_path = None
    best_length = float('inf')
    
    for iteration in range(iterations):
        all_paths = []
        all_lengths = []

        # Each ant constructs a path
        for ant in range(n_ants):
            path = construct_path()
            length = calculate_path_length(path)
            all_paths.append(path)
            all_lengths.append(length)

            # Update best path found so far
            if length < best_length:
                best_path = path
                best_length = length

        # Update pheromone levels based on paths
        update_pheromone(all_paths, all_lengths)

        # Evaporate pheromone
        pheromone * (1 - evaporation_rate)

        print(f"Iteration {iteration + 1}, Best path length: {best_length}")

    return best_path, best_length

# Construct a path for an ant
def construct_path():
    path = [random.randint(0, n_cities - 1)]  # Start from a random city
    visited = set(path)
    
    while len(path) < n_cities:
        current_city = path[-1]
        probabilities = calculate_transition_probabilities(current_city, visited)
        next_city = random.choices(range(n_cities), weights=probabilities)[0]
        path.append(next_city)
        visited.add(next_city)

    path.append(path[0])  # Return to the starting city
    return path

# Calculate the transition probabilities for an ant
def calculate_transition_probabilities(current_city, visited):
    probabilities = []
    total_pheromone = 0.0
    
    for next_city in range(n_cities):
        if next_city not in visited:
            pheromone_strength = pheromone[current_city][next_city] ** alpha
            distance_heuristic = (1.0 / dist_matrix[current_city][next_city]) ** beta
            total_pheromone += pheromone_strength * distance_heuristic

    for next_city in range(n_cities):
        if next_city not in visited:
            pheromone_strength = pheromone[current_city][next_city] ** alpha
            distance_heuristic = (1.0 / dist_matrix[current_city][next_city]) ** beta
            probability = (pheromone_strength * distance_heuristic) / total_pheromone
            probabilities.append(probability)
        else:
            probabilities.append(0)
    
    return probabilities

# Calculate the length of a path
def calculate_path_length(path):
    length = 0.0
    for i in range(len(path) - 1):
        length += dist_matrix[path[i]][path[i + 1]]
    return length

# Update pheromone levels after each iteration
def update_pheromone(all_paths, all_lengths):
    global pheromone
    
    # Evaporate pheromone
    pheromone *= (1 - evaporation_rate)

    for path, length in zip(all_paths, all_lengths):
        deposit = pheromone_deposit / length
        for i in range(len(path) - 1):
            pheromone[path[i]][path[i + 1]] += deposit
            pheromone[path[i + 1]][path[i]] += deposit  # Ensure pheromone is bidirectional

# Run the ACO algorithm
best_path, best_length = aco_algorithm(iterations=100, n_ants=20)

# Display the best path and its length
print("Best path:", best_path)
print("Best path length:", best_length)

# Plotting the best path
x = [cities[i][0] for i in best_path]
y = [cities[i][1] for i in best_path]
plt.plot(x, y, marker='o')
plt.title(f"Best Path (Length: {best_length})")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()
"""