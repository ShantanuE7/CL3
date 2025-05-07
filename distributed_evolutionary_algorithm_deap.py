import random
from deap import base, creator, tools, algorithms
import numpy as np
# Step 1: Define the problem (maximize the Sphere function)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximization problem
creator.create("Individual", list, fitness=creator.FitnessMax)

# Step 2: Create the individual (solution) and population
def create_individual():
    # Generate a list of random values between -5 and 5
    return [random.uniform(-5, 5) for _ in range(10)]  # 10-dimensional problem

# Step 3: Define the evaluation function (Sphere function)
def evaluate(individual):
    return sum(x ** 2 for x in individual),  # Return as a tuple to match DEAP's requirement

# Step 4: Set up the DEAP toolbox
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Step 5: Set up the main function to run the algorithm
def main():
    # Create an initial population of 100 individuals
    population = toolbox.population(n=100)

    # Set up the algorithm parameters
    ngen = 50  # Number of generations
    cxpb = 0.001  # Crossover probability
    mutpb = 0.001  # Mutation probability

    # Run the genetic algorithm using DEAP's algorithms.eaSimple
    result = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, 
                                 ngen=ngen, stats=None, verbose=True)
    
    return population, result

# Run the algorithm
if __name__ == "__main__":
    population, result = main()
    print(population)
    # Display the best individual found
    best_individual = tools.selBest(population, 1)[0]
    print("Best individual:", best_individual)
    print("Fitness:", best_individual.fitness.values)
