import os
import pandas as pd
import random

# Paths
TUN_STOCKS_FILE = './stock market data files/Stock stats/Tunisia Stocks.csv'
PORTFOLIOS_DIR = './Investors portfolios/'

os.makedirs(PORTFOLIOS_DIR, exist_ok=True)

# Load Tunisia Stocks data
tun_df = pd.read_csv(TUN_STOCKS_FILE)

# Extract unique stock names and latest prices
stock_names = tun_df['Name'].unique()

# Create 5 fake investor portfolios
for i in range(1, 6):
    num_stocks = random.randint(3, 5)  # number of stocks per portfolio
    selected_stocks = random.sample(list(stock_names), num_stocks)

    portfolio_data = []
    for stock in selected_stocks:
        # Get latest price for the stock from Tunisia Stocks df
        stock_prices = tun_df[tun_df['Name'] == stock]['Last']
        if stock_prices.empty:
            continue
        last_price = stock_prices.iloc[-1]

        volume = random.randint(5, 10)

        portfolio_data.append({
            'Name': stock,
            'Vol': volume,
            'Price': last_price
        })

    # Save portfolio to CSV
    portfolio_df = pd.DataFrame(portfolio_data)
    portfolio_df.to_csv(f'{PORTFOLIOS_DIR}investor{i}.csv', index=False)
    print(f'Created portfolio: investor{i}.csv')