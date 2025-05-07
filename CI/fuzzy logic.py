def fuzzy_union(A, B):
    return {x: max(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_intersection(A, B):
    return {x: min(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_complement(A):
    return {x: round(1 - μ, 2) for x, μ in A.items()}

def fuzzy_difference(A, B):
    # A - B = A ∩ (Complement of B)
    return fuzzy_intersection(A, fuzzy_complement(B))
def cartesian_product(A, B):
    return {(a, b): round(min(μ_a, μ_b), 2) for a, μ_a in A.items() for b, μ_b in B.items()}

def max_min_composition(R1, R2):
    result = {}
    xs = set(a for (a, b) in R1)
    zs = set(c for (b, c) in R2)

    for x in xs:
        for z in zs:
            min_values = []
            for y in set(b for (a, b) in R1 if a == x):
                μ1 = R1.get((x, y), 0)
                μ2 = R2.get((y, z), 0)
                min_values.append(min(μ1, μ2))
            result[(x, z)] = round(max(min_values), 2) if min_values else 0.0
    return result

A = {'x1': 0.2, 'x2': 0.8, 'x3': 1.0}
B = {'x1': 0.5, 'x2': 0.4, 'x3': 0.7}

print("Union:", fuzzy_union(A, B))
print("Intersection:", fuzzy_intersection(A, B))
print("Complement of A:", fuzzy_complement(A))
print("A - B:", fuzzy_difference(A, B))

# Create fuzzy relations
R1 = cartesian_product(A, B)
R2 = cartesian_product(B, A)

print("Fuzzy Relation R1 (A × B):")
for pair, value in R1.items():
    print(f"{pair}: {value}")

# Max-min composition
print("\nMax-Min Composition of R1 and R2:")
composition = max_min_composition(R1, R2)
for pair, value in composition.items():
    print(f"{pair}: {value}")

'''
Union: {'x1': 0.5, 'x3': 1.0, 'x2': 0.8}
Intersection: {'x1': 0.2, 'x3': 0.7, 'x2': 0.4}
Complement of A: {'x1': 0.8, 'x2': 0.2, 'x3': 0.0}
A - B: {'x1': 0.2, 'x3': 0.3, 'x2': 0.6}
Fuzzy Relation R1 (A × B):
('x1', 'x1'): 0.2
('x1', 'x2'): 0.2
('x1', 'x3'): 0.2
('x2', 'x1'): 0.5
('x2', 'x2'): 0.4
('x2', 'x3'): 0.7
('x3', 'x1'): 0.5
('x3', 'x2'): 0.4
('x3', 'x3'): 0.7

Max-Min Composition of R1 and R2:
('x2', 'x2'): 0.7
('x2', 'x1'): 0.2
('x2', 'x3'): 0.7
('x1', 'x2'): 0.2
('x1', 'x1'): 0.2
('x1', 'x3'): 0.2
('x3', 'x2'): 0.7
('x3', 'x1'): 0.2
('x3', 'x3'): 0.7

'''