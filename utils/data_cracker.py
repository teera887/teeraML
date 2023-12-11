from datetime import timedelta, datetime

import requests
import yfinance as yf
import pandas as pd

import config


def fetch_historical_data(symbol, resolution, api_key):
    try:
        # Define the API endpoint URL for historical data
        historical_url = 'https://finnhub.io/api/v1/stock/candle'

        # Define the time range (from and to) using timestamps
        from_timestamp = int((datetime.today() - timedelta(days=365)).timestamp())
        to_timestamp = int(datetime.today().timestamp())

        # Define parameters for the API request
        params = {
            'symbol': symbol,
            'resolution': resolution,
            'from': from_timestamp,
            'to': to_timestamp,
            'token': api_key
        }

        # Send a GET request to the historical data API
        historical_response = requests.get(historical_url, params=params)

        # Check if the request was successful (status code 200)
        if historical_response.status_code == 200:
            # Parse the JSON response
            historical_data = historical_response.json()

            if 't' in historical_data:
                # Create a DataFrame from the historical data
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(historical_data['t'], unit='s'),
                    'open': historical_data['o'],
                    'high': historical_data['h'],
                    'low': historical_data['l'],
                    'close': historical_data['c'],
                    'volume': historical_data['v']
                })

                return df
            else:
                print(f"No available data for {symbol} at {resolution} resolution.")
                return None

        else:
            print(f"Failed to fetch historical data for {symbol}. Status code: {historical_response.status_code}")
            return None

    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return None


def fetch_quarterly_financial_data(symbol):
    try:
        # Create a Ticker object for the stock
        ticker = yf.Ticker(symbol)

        # Fetch quarterly financial highlights
        financials = ticker.quarterly_financials

        return financials

    except Exception as e:
        print(f"Error fetching quarterly financial data: {e}")
        return None


if __name__ == '__main__':
    symbols = ['NVDA']
    resolutions = ['1', '5', '15', '30', '60', '120', '240', 'D']
    historical_data = fetch_historical_data(symbols[0], resolutions[0], api_key=config.finnhub_api_key)
    financial_data = fetch_quarterly_financial_data(symbols[0])
    pass
