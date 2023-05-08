import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Download historical prices of the S&P 500 index
symbol = "^GSPC"
start_date = "2000-01-01"
end_date = "2023-03-03"
data = yf.download(symbol, start=start_date, end=end_date)

# Calculate log returns
adj_close = data["Adj Close"]
log_returns = np.log(adj_close / adj_close.shift(1)).dropna()

# Calculate monthly volatility
monthly_volatility = log_returns.resample("M").std()

# Calculate realized volatility at month t+1 as a function of the realized volatility at month t
vol_t = monthly_volatility[:-1]
vol_t1 = monthly_volatility[1:]

# Perform linear regression and calculate R-squared value
slope, intercept, r_value, p_value, std_err = linregress(vol_t, vol_t1)
r_squared = r_value**2

# Plot the realized volatility at month t+1 as a function of the realized volatility at month t
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(vol_t, vol_t1, alpha=0.5, label="Data points")
ax.plot([np.min(vol_t), np.max(vol_t)], [np.min(vol_t), np.max(vol_t)], color="red", label="45-degree line")
ax.plot(vol_t, slope*vol_t + intercept, color="green", label=f"Regression line (R²={r_squared:.2f})")
# ax.axvline(x=0.02, color='gray', linestyle='--')
# ax.axvline(x=0.04, color='gray', linestyle='--')
ax.set_xlabel("Realized Volatility at Month t")
ax.set_ylabel("Realized Volatility at Month t+1")
ax.set_title("Volatility Clustering of S&P 500 Index")
ax.legend()
plt.show()
