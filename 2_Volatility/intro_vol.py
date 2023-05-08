import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Define the function
def f(n):
    return np.sqrt(2/(n-1))*gamma(n/2) / gamma((n-1)/2)

# Create an array of x values
x = np.linspace(0.1, 100, 100)

# Calculate the y values
y = f(x)

# Create the plot
plt.plot(x, y)
plt.xlabel('n')
plt.ylabel('h(n)')
plt.title('h(n) as a function of n')

# Display the plot
plt.show()