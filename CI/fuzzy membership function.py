import numpy as np
import matplotlib.pyplot as plt

# Triangular membership function
def triangular_membership(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    else:  # x == b
        return 1

# Generate values and compute membership
a, b, c = 2, 5, 78  # Define triangle parameters
x_vals = np.linspace(4, 100, 200)
mu_vals = [triangular_membership(x, a, b, c) for x in x_vals]

# Plotting
plt.plot(x_vals, mu_vals, label='Triangular MF (a=2, b=5, c=8)', color='blue')
plt.title('Triangular Membership Function')
plt.xlabel('x')
plt.ylabel('Membership Degree')
plt.grid(True)
plt.legend()
plt.show()
