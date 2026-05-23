# =========================================================
# DELTA-ONE EXPECTED SHORTFALL HEDGING SIMULATOR
# =========================================================
#
# Simulates a $1M delta-one position hedged using
# Expected Shortfall (ES) under Basel III conventions.
#
# Main features:
# - 252-day historical simulation for ES estimation
# - 21-day trading simulation
# - Dynamic ES-based hedge rebalancing
# - PnL and hedge effectiveness tracking
#
# Assumptions:
# - Daily volatility: 2%
# - Drift: mu = 0
# - ES confidence level: 97.5%
# =========================================================

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


# Initial parameters
start = time.time()
P0 = 545
old_price = 545
sigma = 0.02
simulation_days = 21
history_days = 252
notional = 1000000
units = notional / P0
old_position = units * P0

# Storage for results
expected_shortfalls = []
hedges = []
pnLs = []
prices_list = []

# ==========================================
# HISTORICAL RISK ENGINE
# ==========================================

# Compute Expected Shortfall from the
# 252-day historical loss distribution

# Simulate historical prices and calculate losses
historical_Z = np.random.normal(0,1,size=history_days)
historical_prices = P0 * np.cumprod(np.exp(sigma * historical_Z))
historical_returns = historical_prices[1:] / historical_prices[:-1] - 1
historical_losses = -notional * historical_returns

# Calculate initial ES
Var = np.quantile(historical_losses, 0.975)
tail_losses = historical_losses[historical_losses > Var]
ES = np.mean(tail_losses)

# Simulate 21 trading days and apply the dynamic hedge
for i in range(simulation_days):
    Zt = np.random.normal(0,1)
    price = old_price*np.exp(sigma*Zt)

    # Calculate the hedge position based on ES and current price
    hedge_position = -ES / price
    
    # Calculate PnL : change in position value + change in hedge value
    new_position = units * price
    PnL = (new_position - old_position) + hedge_position * (old_price - price)

    # Update for next iteration
    old_position = new_position
    old_price = price

    # Store results
    expected_shortfalls.append(ES)
    hedges.append(hedge_position)
    pnLs.append(PnL)
    prices_list.append(price)

# Create DataFrame to display results
DF = pd.DataFrame({
    'Price': prices_list,
    'Expected Shortfall': expected_shortfalls,
    'Hedge': hedges,
    'PnL': pnLs
})
print(DF)


# ==========================================
# VISUALIZATION
# ==========================================


# Plotting results
daily_losses = -DF['PnL']
plt.figure()
plt.plot(
    DF.index,
    daily_losses,
    label='Daily Losses')
plt.axhline(
    y=ES,
    color='r',
    linestyle='--',
    label=f'ES: ${ES:.0f}')
plt.xlabel('Days')
plt.ylabel('Loss ($)')
plt.title('Losses with Expected Shortfall')
plt.legend()
plt.savefig("losses_vs_es.png")
plt.show()

plt.figure()
plt.plot(DF.index, DF['Hedge'])
plt.xlabel('Days')
plt.ylabel('Hedge Position')
plt.title('Dynamic ES Hedge')
plt.savefig("dynamic_hedge.png")
plt.show()

DF['Cumulative_PnL'] = DF['PnL'].cumsum()

plt.figure()
plt.plot(DF.index, DF['Cumulative_PnL'])
plt.xlabel('Days')
plt.ylabel('Cumulative PnL')
plt.title('Cumulative Hedged PnL')
plt.savefig("cumulative_pnl.png")
plt.show()

end = time.time()
print(f"Execution time: {end - start:.6f} seconds")