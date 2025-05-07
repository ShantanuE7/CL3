import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic structural damage dataset
X, y = make_classification(n_samples=200, n_features=6, n_classes=2, random_state=420)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

# Parameters
num_antibodies = 30
num_generations = 50
clone_rate = 3
mutation_rate = 0.1

# Initialize antibodies randomly
antibodies = np.random.rand(num_antibodies, X_train.shape[1])
antibody_labels = np.random.randint(0, 2, size=num_antibodies)

# Affinity = inverse of distance (higher is better)
def affinity(antibody, antigen):
    return 1 / (1 + np.linalg.norm(antibody - antigen))

# Training AIS
for generation in range(num_generations):
    for i, antigen in enumerate(X_train):
        true_label = y_train[i]
        # Compute affinity with all antibodies
        aff = np.array([affinity(ab, antigen) for ab in antibodies])
        # Select best matching antibody
        best_idx = np.argmax(aff)
        if antibody_labels[best_idx] != true_label:
            # Clone and mutate
            clones = np.tile(antibodies[best_idx], (clone_rate, 1))
            noise = np.random.normal(0, mutation_rate, clones.shape)
            clones += noise
            # Update if better
            best_clone = max(clones, key=lambda c: affinity(c, antigen))
            antibodies[best_idx] = best_clone
            antibody_labels[best_idx] = true_label

# Prediction on test set
def predict(antigen):
    aff = [affinity(ab, antigen) for ab in antibodies]
    return antibody_labels[np.argmax(aff)]

y_pred = [predict(x) for x in X_test]

# Accuracy
print("Classification Accuracy:", accuracy_score(y_test, y_pred))
'''
Classification Accuracy: 0.7833333333333333
'''