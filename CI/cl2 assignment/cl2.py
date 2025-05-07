import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import random

# ---------- Load & Preprocess Iris Dataset ----------
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# ---------- Neural Network Builder ----------
def build_model(input_dim, output_dim, layers, neurons, learning_rate):
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))  # Output layer for classification
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------- Genetic Algorithm ----------
def create_population(size):
    return [
        {
            'layers': random.randint(1, 3),
            'neurons': random.randint(4, 64),
            'learning_rate': round(random.uniform(0.0005, 0.01), 4)
        }
        for _ in range(size)
    ]

def fitness(individual):
    model = build_model(X_train.shape[1], y_train.shape[1], individual['layers'], individual['neurons'], individual['learning_rate'])
    model.fit(X_train, y_train, epochs=30, batch_size=8, verbose=0)
    preds = model.predict(X_test)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    acc = accuracy_score(true_classes, pred_classes)
    return acc  # Higher is better

def crossover(p1, p2):
    return {
        key: random.choice([p1[key], p2[key]]) for key in p1
    }

def mutate(individual):
    if random.random() < 0.3:
        individual['layers'] = random.randint(1, 3)
    if random.random() < 0.3:
        individual['neurons'] = random.randint(4, 64)
    if random.random() < 0.3:
        individual['learning_rate'] = round(random.uniform(0.0005, 0.01), 4)
    return individual

def genetic_algorithm(generations=10, population_size=10):
    population = create_population(population_size)

    for generation in range(generations):
        print(f"\nGeneration {generation + 1}")
        scored = [(fitness(ind), ind) for ind in population]
        scored.sort(reverse=True, key=lambda x: x[0])
        print("Best Accuracy:", scored[0][0])
        top = [ind for (_, ind) in scored[:population_size // 2]]

        # Breed next generation
        children = []
        while len(children) < population_size:
            p1, p2 = random.sample(top, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            children.append(child)

        population = children

    best = max([(fitness(ind), ind) for ind in population], key=lambda x: x[0])
    return best

best_score, best_params = genetic_algorithm()
print("\n Best Hyperparameters:", best_params)
print("Best Accuracy on Test Set:", round(best_score * 100, 2), "%")
"""
Best Hyperparameters: {'layers': 3, 'neurons': 4, 'learning_rate': 0.0052}
Best Accuracy on Test Set: 100.0 %
"""