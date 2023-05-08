import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the ticker symbol for the S&P 500
ticker = "^GSPC"

# Set the start and end dates for the historical data
start_date = '2010-01-01'
end_date = '2023-03-03'

# Download the historical data for the S&P 500 from Yahoo Finance
df = yf.download(ticker, start=start_date, end=end_date)

# Calculate the log returns of the S&P 500
df["log_return"] = np.log(df["Close"]) - np.log(df["Close"].shift(1))

# Resample the data to monthly frequency
df_monthly = df.resample("M").last()

# Compute the monthly returns and monthly log returns
df_monthly["monthly_return"] = df_monthly["Close"].pct_change()
df_monthly["monthly_log_return"] = np.log(df_monthly["Close"]) - np.log(df_monthly["Close"].shift(1))

# Remove any rows containing NaN values
df_monthly.dropna(inplace=True)

# Compute the monthly volatility using the standard deviation of monthly log returns
# df_monthly["monthly_volatility"] = df_monthly["monthly_log_return"].std()
df_monthly["monthly_vol"] = df["log_return"].groupby(pd.Grouper(freq='M')).std()

# Define the function to fit
def func(x, a, b, c):
    return a * x**2 + b * x + c

# Fit the function to the data
popt, pcov = curve_fit(func, df_monthly["monthly_log_return"], df_monthly["monthly_vol"])

# Print the optimized parameters
print("a =", popt[0], "b =", popt[1], "c =", popt[2])

# Plot the data and the fit
x = np.linspace(df_monthly["monthly_log_return"].min(), df_monthly["monthly_log_return"].max(), 100)
plt.scatter(df_monthly["monthly_log_return"], df_monthly["monthly_vol"], label="Data")
plt.plot(x, func(x, *popt), 'r-', label="Fit")
plt.xlabel("Monthly Log Return")
plt.ylabel("Monthly Volatility")
plt.legend()
plt.show()
