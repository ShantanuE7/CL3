import numpy as np
import matplotlib.pyplot as plt

# Triangular membership function
def triangular_mf(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return (c - x) / (c - b)

# Generate x values
x_vals = np.linspace(0, 10, 100)
a, b, c = 2, 5, 8  # Define triangle shape
y_vals = [triangular_mf(x, a, b, c) for x in x_vals]

# Plotting
plt.plot(x_vals, y_vals, label='Triangular MF', color='blue')
plt.title("Triangular Membership Function")
plt.xlabel("x")
plt.ylabel("Membership Degree")
plt.grid(True)
plt.legend()
plt.show()
