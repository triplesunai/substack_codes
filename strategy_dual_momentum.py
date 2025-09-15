import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data download
risk_on_asset = 'SPY'       # S&P 500 ETF
alternative_asset = 'TLT'   # 20+ Year Treasury Bond ETF
risk_off_asset = 'SHY'      # 1-3 Year Treasury Bond ETF (our "cash" equivalent)
benchmark = 'SPY'           # Benchmark for comparison
initial_capital = 10000     # Initial capital for backtesting

tickers = [risk_on_asset, alternative_asset, risk_off_asset]

start_date = '2004-01-01'
end_date = pd.to_datetime('today').strftime('%Y-%m-%d')

# Download historical daily data from yahoo finance
prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
print("Data downloaded successfully.")

# Strategy logic
# Lookback for momentum calculation (in months)
relative_momentum_lookback = 6
absolute_momentum_lookback = 10

# Resample to get the last price of each month
monthly_prices = prices.resample('ME').last()

# Calculate momentum (return over the lookback period)
momentum = monthly_prices.pct_change(relative_momentum_lookback).dropna()

# Calculate the long-term moving average for the absolute momentum filter
long_term_ma = monthly_prices.rolling(window=absolute_momentum_lookback).mean().dropna()

# Signal generation
# Align data to ensure we're comparing the same time periods
aligned_momentum = momentum.loc[long_term_ma.index]
aligned_prices = monthly_prices.loc[long_term_ma.index]


signal_df = pd.DataFrame(index=aligned_momentum.index)

# Determine the winner based on relative momentum
signal_df['Winner'] = np.where(
    aligned_momentum[risk_on_asset] > aligned_momentum[alternative_asset],
    risk_on_asset,
    alternative_asset
)

# Apply the absolute momentum filter
def get_signal(row):
    winner = row['Winner']
    price = aligned_prices.loc[row.name, winner]
    ma = long_term_ma.loc[row.name, winner]
    if price > ma:
        return winner
    else:
        return risk_off_asset

signal_df['Signal'] = signal_df.apply(get_signal, axis=1)

# Backtesting
# Create a daily positions df from our monthly signals
positions = signal_df[['Signal']].reindex(prices.index, method='ffill').dropna()
daily_positions = pd.get_dummies(positions['Signal']).reindex(columns=tickers).fillna(0)

daily_returns = prices.pct_change()

strategy_returns = (daily_returns * daily_positions.shift(1)).sum(axis=1)

# Apply transaction costs (5 bp per trade)
transaction_cost = 0.0005
position_changes = daily_positions.diff().abs().sum(axis=1)
transaction_costs = position_changes * transaction_cost
strategy_returns = strategy_returns - transaction_costs

# Performance metrics
def calculate_and_plot_metrics(returns, benchmark_returns, title, position_changes=None):

    print(f"\n--- {title} ---")


    # Total Return
    total_return = ((1 + returns).cumprod().iloc[-1] - 1)

    # Annualized Return (CAGR)
    days = (returns.index[-1] - returns.index[0]).days
    cagr = ((1 + returns).cumprod().iloc[-1]) ** (365.0/days) - 1

    # Annualized volatility
    annualized_volatility = returns.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe_ratio = cagr / annualized_volatility

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative / peak) - 1
    max_drawdown = drawdown.min()

    # Number of trades (position changes)
    num_trades = 0
    if position_changes is not None:
        num_trades = position_changes.sum()
    else:
        # Buy and hold
        num_trades = 2

    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return (CAGR): {cagr:.2%}")
    print(f"Annualized Volatility: {annualized_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Number of Trades: {num_trades:.0f}")

    return (1 + returns).cumprod(), (1 + benchmark_returns).cumprod()

# Calculate metrics for both strategy and benchmark
strategy_cum_returns, benchmark_cum_returns = calculate_and_plot_metrics(
    strategy_returns, daily_returns[benchmark].dropna(), "Strategy Performance", position_changes
)
_ , _ = calculate_and_plot_metrics(
    daily_returns[benchmark].dropna(), daily_returns[benchmark].dropna(), "Benchmark (Buy & Hold SPY) Performance"
)


# Plots
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(15, 8))
(strategy_cum_returns * initial_capital).plot(ax=ax, label='Dual Momentum Strategy')
(benchmark_cum_returns * initial_capital).plot(ax=ax, label='Benchmark (Buy & Hold SPY)', linestyle='--')
ax.set_title('Strategy performance vs. Buy & Hold SPY', fontsize=16)
ax.set_ylabel('Portfolio Value', fontsize=14)
ax.set_xlabel('Date')
ax.legend(loc='upper left')
plt.show()