import streamlit as st
import pandas as pd
import robin_stocks.robinhood as robin
import yfinance as yf
import pyotp
from pandas_datareader import data as web
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pyportfolioopt as pypfopt
import scipy.optimize as sc
from pypfopt import plotting
plt.style.use('fivethirtyeight')

#Title Application
st.write("""
# Robinhood Portfolio Analysis
""")

st.sidebar.header('User Input')

def get_input():
    username = st.sidebar.text_input("Username", "anirudhkamath@berkeley.edu")
    password = st.sidebar.text_input("Password", type="password")
    authenticator = st.sidebar.text_input("Authenticator", "73JD3T6UFTJGJPRS")
    return username, password, authenticator

username,password,authenticator = get_input()

print('Received Password')

#User Login Information
login = robin.login(username, password)
totp = pyotp.TOTP(authenticator).now()

#Gets latest Price for given ticker symbol
def QUOTE(ticker):
    r = robin.get_latest_price(ticker)
    print(ticker.upper() + ": $" + str(r[0]))

#Pulls user portfolio from RH account
my_stocks = robin.build_holdings()
my_profile = robin.build_user_profile()

#Formats portfolio from holdings function into dataframe
tickers = [key[0] for key in my_stocks.items()]
shares = np.array([value[1].get('quantity') for value in my_stocks.items()])
percentage = np.array([value[1].get('percentage') for value in my_stocks.items()])
sector = np.array([yf.Ticker(ticker).info.get('sector') for ticker in tickers])
portfolio = pd.DataFrame({'Stock': tickers, 'Shares': shares, 'Percentage': percentage, 'Sector': sector})

st.header("Current Portfolio")
st.dataframe(portfolio)


#Changes values in dataframe into floats so that we can optimize in later algorithms
equity = float(my_profile.get('equity'))
shares = shares.astype(float)
percentage = percentage.astype(float)/100

#Create a dataframe to plot closing prices of given portfolio
stockStartDate = '2017-01-01'
today = datetime.today().strftime('%Y-%m-%d')
df = pd.DataFrame()
for stock in tickers:
    df[stock] = web.DataReader(stock, data_source = 'yahoo', start = stockStartDate, end = today)['Adj Close']

#Historical Performance of Portfolio
st.line_chart(df)


#Calculate portfolio variance, volatility, and return
returns = df.pct_change()
meanReturns = returns.mean()
cov_matrix_annual = returns.cov() * 252
port_variance = np.dot(percentage.T, np.dot(cov_matrix_annual, percentage))
port_volatility = np.sqrt(port_variance)
portfolioSimpleAnnualReturn = (np.sum(returns.mean() * percentage) * 252)

#Portfolio Correlation Matrix

st.header("Correlation Matrix")

f = plt.figure(figsize=(5, 5))
plt.matshow(returns.corr(), fignum=f.number)
plt.xticks(range(returns.select_dtypes(['number']).shape[1]), returns.select_dtypes(['number']).columns, fontsize=7, rotation=45)
plt.yticks(range(returns.select_dtypes(['number']).shape[1]), returns.select_dtypes(['number']).columns, fontsize=7)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=7)
plt.title('Correlation Matrix', fontsize=8);

st.pyplot(f)

#Format Portfolio Statistics
percent_var = str(round(port_variance, 2)*100) + '%'
percent_vols = str(round(port_volatility, 2)*100) + '%'
percent_ret = str(round(portfolioSimpleAnnualReturn, 2)*100) + '%'
sharpe_ratio = str(round(((portfolioSimpleAnnualReturn*100-2.02)/(port_volatility*100)), 2))

st.text('Expected annual return: ' + percent_ret)
st.text('Annual volatility: ' + percent_vols)
st.text('Sharpe Ratio: ' + sharpe_ratio)

#Optimize Portfolio Weights
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

fig, ax = plt.subplots()

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ret_tangent, std_tangent, _ = ef.portfolio_performance(verbose = True)

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = equity)
allocation, leftover = da.lp_portfolio()
print('Discrete Allocation:', allocation)
print('Funds remaining: ${:.2f}'.format(leftover))


st.header( "Optimal Allocation\n")
allocation_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
st.dataframe(allocation_df)

st.text('Funds remaining: ${:.2f}'.format(leftover))

st.text('Expected annual return: ' + str(round(ef.portfolio_performance()[0], 2)*100) + "%")
st.text('Annual volatility: ' + str(round(ef.portfolio_performance()[1], 2)*100) + "%")
st.text('Sharpe Ratio: ' + str(round(ef.portfolio_performance()[2], 2)))

ef = EfficientFrontier(mu, S)
fig, ax = plt.subplots()

# Find the tangency portfolio
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe", zorder=2)
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True, zorder=3)

# Generate random portfolios
n_samples = 10000
w = np.random.dirichlet(np.ones(len(mu)), n_samples)
rets = w.dot(mu)
stds = np.sqrt(np.diag(w @ S @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)

st.pyplot(fig)

