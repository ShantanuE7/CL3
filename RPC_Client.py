import xmlrpc.client

# Connect to the server
proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")

# Take input from user
num = int(input("Enter an integer to calculate factorial: "))

# Call the remote function
result = proxy.compute_factorial(num)

# Print the result
print(f"Factorial of {num} is {result}")










