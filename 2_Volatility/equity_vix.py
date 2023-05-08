import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Download the data from Yahoo Finance
start_date = '2000-01-01'
end_date = '2023-03-03'
vix = yf.download('^VIX', start=start_date, end=end_date)['Adj Close']
spx = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']

# Calculate the daily returns
daily_returns_vix = vix.pct_change()
daily_returns_spx = spx.pct_change()

# Define the quadratic function
def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

# Fit the quadratic function to the data
popt, pcov = curve_fit(quadratic, daily_returns_spx.dropna(), daily_returns_vix.dropna())

# Generate the curve
x = np.linspace(-0.1, 0.1, 100)
y = quadratic(x, *popt)

# Plot the data and the curve
plt.scatter(daily_returns_spx, daily_returns_vix, label="Data")
plt.plot(x, y, color='red', label="Fit")
plt.xlabel('S&P 500 daily returns')
plt.ylabel('VIX daily returns')
plt.title('Comparison of daily returns between VIX and S&P 500')
plt.legend()
plt.show()
