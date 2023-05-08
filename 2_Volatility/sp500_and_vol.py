import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
# Download historical prices of the S&P 500 index
import yfinance as yf

symbol = "^GSPC"
start_date = "2000-01-01"
end_date = "2023-03-03"

df = yf.download(symbol, start=start_date, end=end_date)

# Calculate daily returns
returns = df["Adj Close"].pct_change().dropna()

# Calculate rolling standard deviation of returns with a 30-day window
rolling_sd = returns.rolling(window=30).std()

# Plot the S&P 500 index and the rolling standard deviation of returns
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["Adj Close"])
ax.set_ylabel("Price")
ax.set_title("S&P 500 Index with Rolling Standard Deviation of Returns")
ax2 = ax.twinx()
ax2.plot(rolling_sd, color="orange")
ax2.set_ylabel("Standard Deviation")
plt.show()
