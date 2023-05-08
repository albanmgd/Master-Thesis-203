import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch S&P 500 index data from Yahoo Finance
sp500 = yf.download('^GSPC', start='2000-01-01', end='2023-03-03', progress=False)['Adj Close']

# Compute daily log returns
log_returns = np.log(sp500 / sp500.shift(1)).dropna()

# Compute monthly realized volatility as the standard deviation of daily log returns within each month
vol = log_returns.groupby(pd.Grouper(freq='M')).std()

# Determine the highest and lowest 10% of monthly volatility
top_10_threshold = vol.quantile(0.9)
bottom_10_threshold = vol.quantile(0.1)

# Find the dates where the monthly volatility is in the top or bottom 10%
top_10_dates = vol[vol >= top_10_threshold].index
bottom_10_dates = vol[vol <= bottom_10_threshold].index

# Calculate the mean monthly volatility for each bucket over time
top_10_vol = []
bottom_10_vol = []

for i in range(0, 10):
    top_index = [vol.index.get_loc(date) + i for date in top_10_dates]
    bottom_index = [vol.index.get_loc(date) + i for date in bottom_10_dates]

    bucket_highest = [vol[top_index]]
    bucket_lowest = [vol[bottom_index]]
    top_10_vol.append(np.mean(bucket_highest))
    bottom_10_vol.append(np.mean(bucket_lowest))
    print("done")

top_10_months = [i for i in range(0, 10)]

# Plot the volatility of each bucket against the number of months after the initial observation
plt.figure(figsize=(10, 6))
plt.plot(top_10_months, top_10_vol, '', label='Top 10% Volatility')
plt.plot(top_10_months, bottom_10_vol, '', label='Bottom 10% Volatility')
plt.axhline(y=vol.mean(), color='black', linestyle='--', label='Long-Term Mean')
plt.title('S&P 500 Volatility Buckets Mean Reversion')
plt.xlabel('Months After Initial Observation')
plt.ylabel('Monthly Realized Volatility')
plt.legend()
plt.show()
