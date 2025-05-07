import random
from collections import defaultdict

class LoadBalancer:
    def __init__(self, servers, algorithm='round_robin'):
        self.servers = servers
        self.algorithm = algorithm
        self.counter = 0  # for round robin
        self.connections = defaultdict(int)  # for least connections

    def get_server(self):
        if self.algorithm == 'round_robin':
            server = self.servers[self.counter % len(self.servers)]
            self.counter += 1
            return server

        elif self.algorithm == 'least_connections':
            return min(self.servers, key=lambda s: self.connections[s])

        elif self.algorithm == 'random':
            return random.choice(self.servers)

    def handle_request(self, request_id):
        server = self.get_server()
        self.connections[server] += 1
        print(f"Request {request_id} assigned to {server}")

    def release_connection(self, server):
        if self.connections[server] > 0:
            self.connections[server] -= 1

# --- Simulation Code ---
servers = ['Server-1', 'Server-2', 'Server-3']
lb = LoadBalancer(servers, algorithm='least_connections')  # Change to 'round_robin' or 'random'

# Simulate 10 client requests
for i in range(10):
    lb.handle_request(f"Client {i+1}")

# Optional: simulate releasing some connections
lb.release_connection('Server-1')
lb.release_connection('Server-2')


#-------------------------------------------------------------------------------------------------------
"""
lb = LoadBalancer(servers, algorithm='round_robin')  # or 'random'
"""
#-------------------------------------------------------------------------------------------------------