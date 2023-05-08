import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values
x = np.linspace(-5, 5, 100)


# Define the squared loss function
def squared_loss(y_true, y_pred):
    return np.square(y_true - y_pred)


# Define the absolute value loss function
def absolute_loss(y_true, y_pred):
    return np.abs(y_true - y_pred)


# Define the Huber loss function
def huber_loss(y_true, y_pred, delta=1.0):
    abs_diff = np.abs(y_true - y_pred)
    if abs_diff <= delta:
        return 0.5 * np.square(abs_diff)
    else:
        return delta * (abs_diff - 0.5 * delta)


# Calculate the losses for each x value
squared_losses = squared_loss(0, x)
absolute_losses = absolute_loss(0, x)
huber_losses_delta_1 = [huber_loss(0, xi) for xi in x]
huber_losses_delta_1_2 = [huber_loss(0, xi, 0.5) for xi in x]

# Plot the losses
plt.plot(x, squared_losses, label='Squared Loss')
plt.plot(x, absolute_losses, label='Absolute Loss')
plt.plot(x, huber_losses_delta_1, label='Huber Loss delta=1')
plt.plot(x, huber_losses_delta_1_2, label='Huber Loss delta=0.5')
plt.legend()
plt.title("Loss values against y - f(x)")
plt.xlabel('y - f(x)')
plt.ylabel('Loss')
plt.show()
