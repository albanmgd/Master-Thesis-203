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
# Calculate the long-term average of rolling standard deviation
long_term_mean = monthly_volatility.mean()

# Plot rolling standard deviation and long-term average
plt.figure(figsize=(10, 6))
plt.plot(monthly_volatility, label='Monthly standard deviation')
plt.axhline(y=long_term_mean, color='r', linestyle='--', label='Long-term mean')
plt.title('S&P 500 Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()