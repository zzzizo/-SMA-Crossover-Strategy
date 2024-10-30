import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch historical data for SPY (S&P 500 ETF)
ticker = "SPY"
data = yf.download(ticker, start="2018-01-01", end="2023-01-01")

# Calculate the short-term and long-term SMAs
short_window = 20  # Short-term moving average (e.g., 20 days)
long_window = 50   # Long-term moving average (e.g., 50 days)

data['SMA_20'] = data['Close'].rolling(window=short_window).mean()
data['SMA_50'] = data['Close'].rolling(window=long_window).mean()

# Generate Buy/Sell signals - use `iloc` to avoid errors with positional indexing
data['Signal'] = 0
data.iloc[short_window:, data.columns.get_loc('Signal')] = np.where(
    data['SMA_20'][short_window:] > data['SMA_50'][short_window:], 1, -1
)

# Calculate returns
data['Return'] = data['Close'].pct_change()
data['Strategy_Return'] = data['Return'] * data['Signal'].shift(1)

# Calculate performance metrics
cumulative_return = (1 + data['Strategy_Return']).cumprod() - 1
drawdown = (1 + data['Strategy_Return']).cumprod().div((1 + data['Strategy_Return']).cumprod().cummax()) - 1
max_drawdown = drawdown.min()
win_rate = (data['Strategy_Return'] > 0).mean()

# Display the basic metrics
# Using .iloc[-1] to access the last value of cumulative_return for compatibility
print(f"Cumulative Return: {cumulative_return.iloc[-1]:.2%}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Win Rate: {win_rate:.2%}")

# Plot the strategy performance
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label=f'{ticker} Price', color='black', alpha=0.5)
plt.plot(data['SMA_20'], label=f'{short_window}-Day SMA', color='blue', linestyle='--')
plt.plot(data['SMA_50'], label=f'{long_window}-Day SMA', color='red', linestyle='--')

# Mark Buy and Sell signals on the plot
plt.plot(data[data['Signal'] == 1].index, data['Close'][data['Signal'] == 1], '^', color='green', label='Buy Signal', markersize=8)
plt.plot(data[data['Signal'] == -1].index, data['Close'][data['Signal'] == -1], 'v', color='red', label='Sell Signal', markersize=8)

plt.title(f'Simple Moving Average Crossover Strategy for {ticker}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='best')
plt.show()
