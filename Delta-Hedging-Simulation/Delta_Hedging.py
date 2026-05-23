# Objective: Simulate a dynamic delta-hedging strategy for a vanilla equity call option using Python, computing daily PnL attribution and risk metrics to optimize trading performance.
# Deliverables:
# A time series of daily stock positions, option values, and PnL.
# Risk metrics: delta, gamma, vega at each step.#
#  A summary of total PnL and hedging effectiveness.

import pandas as pd
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt


# ========================
# DAY 0 - INITIALIZATION
# ========================

# Parameters
sigma = 0.2
r = 0.02
q = 0
S = 100
K = 100
T = 1
Shares = 10000


# Black-Scholes formula
D1 = ((np.log(S/K) + (r+sigma**2/2)*T))/(sigma*np.sqrt(T))
D2 = D1 - sigma*np.sqrt(T)
nd1 = np.exp(-D1**2 / 2) / np.sqrt(2 * np.pi)
Call_Price = S*np.exp(-q*T)*stats.NormalDist().cdf(D1) - K*np.exp(-r*T)*stats.NormalDist().cdf(D2)
Delta = np.exp(-q*T)*stats.NormalDist().cdf(D1)
Gamma = nd1/(S*sigma*np.sqrt(T))
Hedging = (-Delta * Shares)


# Display the call price, delta and hedging strategy
print(f"Call price: {round(Call_Price,3)}")
if Hedging > 0:
    print(f"Hedge: Buy {Hedging:.2f} shares")
elif Hedging < 0:
    print(f"Hedge: Sell {abs(Hedging):.2f} shares")
else:
    print("Hedge: None")


# ========================
# DAY 1-21 - IMPLEMENTING THE HEDGING STRATEGY
# ========================

# Black-Scholes formula to calculate the price, delta, gamma and vega of the call option

def bs_greeks(S, K, r, T, sigma):
    D1 = ((np.log(S/K) + (r+sigma**2/2)*T))/(sigma*np.sqrt(T))
    D2 = D1 - sigma*np.sqrt(T)
    nd1 = np.exp(-D1**2 / 2) / np.sqrt(2 * np.pi)
    Price = S*np.exp(-q*T)*stats.NormalDist().cdf(D1) - K*np.exp(-r*T)*stats.NormalDist().cdf(D2)
    Delta = np.exp(-q*T)*stats.NormalDist().cdf(D1)
    Gamma = nd1/(S*sigma*np.sqrt(T))
    Vega = S*np.sqrt(T)*nd1
    return Price, Delta, Gamma, Vega

# Initialize with old values
Stock_old = S
Price_old = Call_Price
Delta_old = Delta
Hedging_old = Hedging

# List-creation for storing
Stocks = []
Prices = []
Deltas = []
Pnls = []  
Rebalances = []       
Gammas = []
Vegas = []

# Simulate the stock price for 21 days and calculate the price, delta, pnl and rebalance at each day

for i in range(1,22):
    T=T-1/252
    Zt = np.random.normal(0, 1)
    Stock = Stock_old*np.exp(((r - 0.5 * sigma**2))*(1/252)+sigma*np.sqrt((1/252))*Zt)
    Price, Delta, Gamma, Vega = bs_greeks(Stock, K, r, T, sigma)
    Pnl = (Price - Price_old) + Delta_old * (Stock_old - Stock) + r * (Delta_old * Stock_old - Price_old) / 252
    Hedging = (-Delta * Shares)
    Rebalance = (Hedging - Hedging_old) * Stock
    Stock_old = Stock
    Price_old = Price
    Delta_old = Delta
    Hedging_old = Hedging
    Stocks.append(Stock)
    Prices.append(Price)
    Deltas.append(Delta)
    Gammas.append(Gamma)
    Vegas.append(Vega)
    Pnls.append(Pnl)
    Rebalances.append(Rebalance)

# Display the results in a Dataframe and plot the Pnl vs Stock price
DF = pd.DataFrame({
'Stock': Stocks,
'Price': Prices,
'Delta': Deltas,
'Gamma': Gammas,
'Vega': Vegas,
'PnL': Pnls,
'Rebalance': Rebalances
})
DF.index = range(1,22)
print(DF)
# Plot 1 : PnL vs Stock
DF_sorted = DF.sort_values(by='Stock')

plt.figure()
plt.plot(DF_sorted['Stock'], DF_sorted['Pnl'])
plt.xlabel('Stock Price')
plt.ylabel('PnL')
plt.title('PnL vs Stock')
plt.show()


# Plot 2 : Cumulative PnL
DF['CumPnL'] = DF['Pnl'].cumsum()

plt.figure()
plt.plot(DF.index, DF['CumPnL'])
plt.xlabel('Time (days)')
plt.ylabel('Cumulative PnL')
plt.title('Cumulative PnL')
plt.show()

    








