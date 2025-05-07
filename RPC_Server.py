from xmlrpc.server import SimpleXMLRPCServer

# Factorial Function
def factorial(n):
    if n < 0:
        return "Error: Negative numbers not allowed"
    elif n == 0 or n == 1:
        return 1
    else:
        fact = 1
        for i in range(2, n + 1):
            fact *= i
        return fact

# Set up the RPC Server
server = SimpleXMLRPCServer(("localhost", 8000))
print("Server is listening on port 8000...")

# Register the function
server.register_function(factorial, "compute_factorial")

# Run the server
server.serve_forever()
