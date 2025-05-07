import numpy as np

# Define fuzzy sets as dictionaries
A = {'x1': 0.2, 'x2': 0.7, 'x3': 0.5}
B = {'x1': 0.6, 'x2': 0.4, 'x3': 0.8}

universe = A.keys()

# Union (max)
def fuzzy_union(A, B):
    return {x: max(A[x], B[x]) for x in universe}

# Intersection (min)
def fuzzy_intersection(A, B):
    return {x: min(A[x], B[x]) for x in universe}

# Complement (1 - membership)
def fuzzy_complement(A):
    return {x: round(1 - A[x], 2) for x in universe}

# Difference (A - B): min(A(x), 1 - B(x))
def fuzzy_difference(A, B):
    return {x: min(A[x], 1 - B[x]) for x in universe}

# Cartesian product (fuzzy relation)
def cartesian_product(A, B):
    return {(x, y): min(A[x], B[y]) for x in A for y in B}

# Max-min composition of fuzzy relations R1 and R2
def max_min_composition(R1, R2):
    domain_x = set(i for i, _ in R1)
    domain_z = set(j for _, j in R2)
    composition = {}
    for x in domain_x:
        for z in domain_z:
            mins = [min(R1.get((x, y), 0), R2.get((y, z), 0)) for y in set(j for _, j in R1)]
            composition[(x, z)] = max(mins) if mins else 0
    return composition

# ========== Outputs ========== #

print("Union:", fuzzy_union(A, B))
print("Intersection:", fuzzy_intersection(A, B))
print("Complement of A:", fuzzy_complement(A))
print("Difference A - B:", fuzzy_difference(A, B))

# Fuzzy relation via Cartesian product
R1 = cartesian_product(A, B)
R2 = cartesian_product(B, A)

print("\nFuzzy Relation R1 (A x B):", R1)
print("Fuzzy Relation R2 (B x A):", R2)

# Max-Min Composition of two relations
composition = max_min_composition(R1, R2)
print("\nMax-Min Composition (R1 o R2):", composition)
