import numpy as np
import matplotlib.pyplot as plt

# ----- Distance Matrix -----
# Example: coordinates of 10 cities (can be changed)
city_coords = np.array([
    [0, 0],
    [1, 5],
    [5, 2],
    [6, 6],
    [8, 3],
    [7, 0],
    [2, 7],
    [3, 3],
    [4, 9],
    [9, 6]
])

# Compute distance matrix
def distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(coords[i] - coords[j])
    return dist

# ----- Ant Colony Optimization -----
class ACO:
    def __init__(self, n_ants, n_best, n_iter, decay, alpha=1, beta=2):
        self.n_ants = n_ants            # Number of ants per iteration
        self.n_best = n_best            # Best ants to update pheromones
        self.n_iter = n_iter            # Iterations
        self.decay = decay              # Pheromone decay rate
        self.alpha = alpha              # Pheromone importance
        self.beta = beta                # Distance importance

    def run(self, dist_matrix):
        n_cities = len(dist_matrix)
        pheromone = np.ones((n_cities, n_cities))
        all_time_best = ([], np.inf)

        for iteration in range(self.n_iter):
            all_routes = self.construct_solutions(pheromone, dist_matrix)
            self.spread_pheromone(pheromone, all_routes, self.n_best)
            pheromone *= self.decay  # Evaporation
            best_route = min(all_routes, key=lambda x: x[1])
            if best_route[1] < all_time_best[1]:
                all_time_best = best_route
            if iteration%10==0:    
                print(f"Iteration {iteration}, Best Route Length: {best_route[1]:.2f}")

        return all_time_best

    def construct_solutions(self, pheromone, dist_matrix):
        all_routes = []
        for _ in range(self.n_ants):
            route = self.generate_route(pheromone, dist_matrix)
            all_routes.append((route, self.route_distance(route, dist_matrix)))
        return all_routes

    def generate_route(self, pheromone, dist_matrix):
        n = len(dist_matrix)
        route = []
        visited = set()
        current = np.random.randint(n)
        route.append(current)
        visited.add(current)

        for _ in range(n - 1):
            probs = self.probabilities(current, visited, pheromone, dist_matrix)
            next_city = self.select_next_city(probs)
            route.append(next_city)
            visited.add(next_city)
            current = next_city

        return route

    def probabilities(self, current, visited, pheromone, dist_matrix):
        n = len(dist_matrix)
        probs = np.zeros(n)
        for j in range(n):
            if j not in visited:
                pher = pheromone[current][j] ** self.alpha
                heuristic = (1 / dist_matrix[current][j]) ** self.beta
                probs[j] = pher * heuristic
        probs /= probs.sum()
        return probs

    def select_next_city(self, probs):
        return np.random.choice(len(probs), p=probs)

    def route_distance(self, route, dist_matrix):
        distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            distance += dist_matrix[from_city][to_city]
        return distance

    def spread_pheromone(self, pheromone, all_routes, n_best):
        sorted_routes = sorted(all_routes, key=lambda x: x[1])
        for route, dist in sorted_routes[:n_best]:
            for i in range(len(route)):
                from_city = route[i]
                to_city = route[(i + 1) % len(route)]
                pheromone[from_city][to_city] += 1.0 / dist
                pheromone[to_city][from_city] += 1.0 / dist

# ----- Run ACO -----
coords = city_coords
dist_mat = distance_matrix(coords)

aco = ACO(n_ants=20, n_best=5, n_iter=101, decay=0.95, alpha=1, beta=3)
best_route, best_length = aco.run(dist_mat)

# ----- Plot the Best Route -----
def plot_route(route, coords):
    route = route + [route[0]]  # complete the loop
    x = [coords[i][0] for i in route]
    y = [coords[i][1] for i in route]
    plt.plot(x, y, marker='o', linestyle='-')
    for idx, (x_i, y_i) in enumerate(coords):
        plt.text(x_i + 0.2, y_i + 0.2, str(idx), fontsize=12)
    plt.title("Best TSP Route Found by ACO")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

plot_route(best_route, coords)
'''
Iteration 0, Best Route Length: 37.62
Iteration 10, Best Route Length: 35.16
Iteration 20, Best Route Length: 32.40
Iteration 30, Best Route Length: 32.40
Iteration 40, Best Route Length: 32.40
Iteration 50, Best Route Length: 32.40
Iteration 60, Best Route Length: 32.40
Iteration 70, Best Route Length: 32.40
Iteration 80, Best Route Length: 32.40
Iteration 90, Best Route Length: 32.40
Iteration 100, Best Route Length: 32.40

'''