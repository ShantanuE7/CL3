import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,r2_score

# Dummy data banavlay: 100 examples ahet, pratek madhe 5 features
X = np.random.rand(300, 3)  # 3 features (e.g., inlet temp, feed rate, etc.)
y = np.sin(X[:, 0]) + np.log1p(X[:, 1]) + X[:, 2]**2 

# He function check karto ki model chi prediction kiti exact ahe
def fitness(model, X, y):
    preds = model.predict(X)
    return mean_squared_error(y, preds)  # jast error mhnje model khota boltoy 😅

# He function model cha weight thoda random badalto - mhnje mutation
def mutate(model):
    for i in range(len(model.coefs_)):
        model.coefs_[i] += 0.01 * np.random.randn(*model.coefs_[i].shape)
    for i in range(len(model.intercepts_)):
        model.intercepts_[i] += 0.01 * np.random.randn(*model.intercepts_[i].shape)
    return model

# Ata kharach Genetic Algorithm suru karuya
def simple_genetic_algorithm(X, y):
    population = []  # Saglyat pahile 5 model create karayche (mhnje 1st generation)
    for _ in range(5):
        model = MLPRegressor(hidden_layer_sizes=(5,), max_iter=500)
        model.fit(X, y)  # pahilyanda data war train karun ghetlay
        population.append(model)

    # Ata apan teen generation chalavnar
    for gen in range(3):
        print(f"Generation {gen+1} chalu aahe...")

        # pratek model chi performance check karaychi
        fitness_scores = [fitness(m, X, y) for m in population]

        # je model best perform karto (minimum error), te ghetlay
        best_model = population[np.argmin(fitness_scores)]
        print("Best score (kam error):", min(fitness_scores))

        # navi generation tayar karaychi - sagle mutate karun tayar hoil
        population = [mutate(best_model) for _ in range(5)]

    return best_model  # best model return karto

# Ata main part suru karto
best_nn = simple_genetic_algorithm(X, y)

# Final prediction karaychya
preds = best_nn.predict(X)
# print("Predictions (first 10 values bagh):", preds[:10])
print("\n=== Final Evaluation ===")
print("MSE:", mean_squared_error(y, preds))
print("R2:", r2_score(y, preds))
































































































































"""

from sklearn.neural_network import MLPRegressor
import numpy as np
import random

# Fuzzy Fitness Function for Optimization
def fitness_function(nn, X, y):
    predictions = nn.predict(X)  # Predict with the already trained model
    return np.mean((predictions - y)**2)  # Mean Squared Error

# Genetic Algorithm to optimize Neural Network
def genetic_algorithm(nn, X, y, population_size=20, generations=50, mutation_rate=0.1):
    population = [nn for _ in range(population_size)]  # Initial population of NN models
    best_nn = nn
    best_fitness = float('inf')
    
    for gen in range(generations):
        fitness_values = [fitness_function(individual, X, y) for individual in population]
        best_idx = np.argmin(fitness_values)
        
        # Store the best performing NN and its fitness
        if fitness_values[best_idx] < best_fitness:
            best_nn = population[best_idx]
            best_fitness = fitness_values[best_idx]
        
        # Crossover & Mutation
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(population, 2)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.extend([child1, child2])
        
        population = [mutate(nn, mutation_rate) for nn in new_population]
    
    return best_nn

# Placeholder functions for crossover and mutation
def crossover(parent1, parent2):
    # Implement crossover (combining weights and biases)
    return parent1

def mutate(nn, mutation_rate):
    # Randomly mutate the neural network parameters (weights)
    return nn

# Main function to test the GA-NN model for spray drying
if __name__ == "__main__":
    # Example data for spray drying (replace with actual data)
    X = np.random.rand(100, 5)  # Features: inlet air temp, flow rate, etc.
    y = np.random.rand(100)  # Output: quality measures like moisture content

    # Neural Network Setup
    nn = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000)
    
    # **Initial training** of the neural network
    nn.fit(X, y)  # Fit the model with the data first
    
    # Genetic Algorithm Optimization
    optimized_nn = genetic_algorithm(nn, X, y)
    
    # Train with optimized NN
    optimized_nn.fit(X, y)
    predictions = optimized_nn.predict(X)
    print(predictions)
"""