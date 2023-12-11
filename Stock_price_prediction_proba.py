import os
import time
import warnings
from datetime import timedelta, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import retry
import ta
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense


# Suppress RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define your API keys
finnhub_api_key = 'cli3mlpr01qh8ugjfumgcli3mlpr01qh8ugjfun0'

num_days_to_predict = 1

# Define your output directory
output_dir = '/Users/teera/Downloads/'

# Define the historical data window (replace with your desired window size)
historical_data_window = 30

# Define the feature columns for prediction
feature_columns = [
    'open', 'high', 'low', 'close', 'volume',
    'rsi', 'stoch_k', 'cci', 'adx', 'ao',
    'momentum_10', 'macd', 'stoch_rsi', 'wpr',
    'uo', 'mfi', 'force_index', 'bb_upper',
    'bb_middle', 'bb_lower'
]


def fetch_historical_data(symbol, resolution, api_key, prediction_period):
    try:
        # Define the API endpoint URL for historical data
        historical_url = 'https://finnhub.io/api/v1/stock/candle'

        # Define the time range (from and to) using timestamps
        from_timestamp = int((datetime.today() - timedelta(days=5)).timestamp())
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

                # # Shift the 'close' column for each future day to create the target variable
                # for nthDay in range(1, num_days_to_predict + 1):
                #     df[f'next_{nthDay}day_future_close'] = df['close'].shift(-nthDay)
                #     df[f'next_{nthDay}day'] = df['timestamp'] + pd.to_timedelta(nthDay, unit='D')

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


def calculate_technical_probabilities(df):
    probabilities = {}

    # Define indicators to calculate probabilities
    indicators = [
        ('rsi', 70),
        ('stoch_k', 80),
        ('cci', 100),
        ('adx', 25),
        ('ao', 0),
        ('momentum_10', 0),
        ('macd', 0),
        ('stoch_rsi', 0.8),
        ('wpr', -20),
        ('uo', 50),
        ('mfi', 80),
        ('force_index', 0)
    ]

    for indicator, threshold in indicators:
        try:
            indicator_values = df[indicator].dropna()

            if len(indicator_values) > 0:
                indicator_probability = len(indicator_values[indicator_values > threshold]) / len(indicator_values)
                probabilities[indicator] = indicator_probability
            else:
                probabilities[indicator] = 0  # Default to neutral if no data

        except Exception as e:
            print(f"Error calculating {indicator} probability: {e}")
            probabilities[indicator] = 0  # Default to neutral

        # Calculate Bollinger Bands probabilities
        try:
            bb_values = df[['bb_upper', 'bb_middle', 'bb_lower']].dropna()

            if len(bb_values) > 0:
                bb_buy_probability = len(bb_values[(bb_values['bb_upper'] > 0) & (bb_values['bb_lower'] > 0)]) / len(
                    bb_values)
                bb_sell_probability = len(bb_values[(bb_values['bb_upper'] <= 0) | (bb_values['bb_lower'] <= 0)]) / len(
                    bb_values)

                probabilities['bollinger_middle'] = bb_buy_probability
                probabilities['bollinger_upper'] = bb_buy_probability
                probabilities['bollinger_lower'] = bb_sell_probability
            else:
                probabilities['bollinger_middle'] = 0
                probabilities['bollinger_upper'] = 0
                probabilities['bollinger_lower'] = 0

        except Exception as e:
            print(f"Error calculating Bollinger Bands probabilities: {e}")
            probabilities['bollinger_middle'] = 0
            probabilities['bollinger_upper'] = 0
            probabilities['bollinger_lower'] = 0

        # # Calculate Fibonacci retracement levels probabilities
        # try:
        #     fibonacci_0_382_probability = len(df[df['fib_0.382'] > 0]) / len(df)
        #     fibonacci_0_618_probability = len(df[df['fib_0.618'] > 0]) / len(df)

        #     probabilities['fib_0.382'] = fibonacci_0_382_probability
        #     probabilities['fib_0.618'] = fibonacci_0_618_probability

        # except Exception as e:
        #     print(f"Error calculating Fibonacci levels probabilities: {e}")
        #     probabilities['fib_0.382'] = 0
        #     probabilities['fib_0.618'] = 0

    return probabilities



# Function to make predictions using LSTM model
def make_lstm_predictions(model, X_valid_scaled):
    predictions = model.predict(X_valid_scaled)
    return predictions


# Function to update dataframe with LSTM predictions
def update_dataframe_with_lstm_prediction(df, predictions):
    df['predicted_future_close'] = predictions.flatten()
    return df


def train_and_save_lstm_models(data_dict, feature_columns, target_column, prediction_period):
    for symbol, df in data_dict.items():
        try:
            print(f"Training model for symbol: {symbol}")
            X_train_scaled, X_valid_scaled, y_train, y_valid, scaler = load_and_split_dataset(df, feature_columns,
                                                                                              target_column)

            # Reset the LSTM model for each symbol
            input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
            model = create_lstm_model(input_shape)

            # Train the LSTM model
            model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_valid_scaled, y_valid),
                      verbose=1)

            if model:
                # Save the trained model to a file with the symbol name
                model_filename = os.path.join(output_dir, f'{symbol}_lstm_model.h5')
                model.save(model_filename)
                print(f"Saved LSTM model for {symbol} to {model_filename}")

                # Evaluate the model on the validation set
                loss = model.evaluate(X_valid_scaled, y_valid)
                print(f"Validation Loss for {symbol}: {loss}")

                # Make predictions using the trained LSTM model
                predictions = make_lstm_predictions(model, X_valid_scaled)

                if len(predictions) > 0:
                    # Update the DataFrame with the predicted prices
                    df['predicted_future_close'] = predictions.flatten()

        except Exception as e:
            print(f"Error training model for {symbol}: {e}")


def prepare_input_features_lstm(df, feature_columns, target_column, window_size=10, prediction_days=1):
    features = df[feature_columns].values
    target = df[target_column].values
    input_feature_vectors = []
    target_values = []

    for i in range(window_size, len(features) - prediction_days + 1):
        window_data = features[i - window_size: i]
        target_value = target[i - 1:i - 1 + prediction_days]  # Adjusted to include multiple prediction days
        input_feature_vectors.append(window_data)
        target_values.append(target_value)

    return np.array(input_feature_vectors), np.array(target_values)


def fetch_and_process_symbol_data(symbol, resolutions, finnhub_api_key, prediction_period, num_days_to_predict,
                                  feature_columns):
    # Initialize bullish_percentage and bearish_percentage
    bullish_percentage = 0
    bearish_percentage = 0

    # Initialize lists to store data
    historical_data_list = []
    indicators_and_probabilities_list = []

    for resolution in resolutions:
        print(f"Resolution: {resolution}")  # Print resolution label

        # Predict future prices and update the DataFrame
        df = fetch_historical_data(symbol, resolution, finnhub_api_key, prediction_period)

        if df is None:
            continue  # Move to the next resolution

        try:

            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()

            # Stochastic %K
            df['stoch_k'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()

            # CCI
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])

            # ADX
            df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()

            # Awesome Oscillator
            df['ao'] = ta.momentum.AwesomeOscillatorIndicator(df['high'], df['low']).awesome_oscillator()

            # Momentum (10)
            df['momentum_1'] = ta.momentum.roc(df['close'], 1)

            # MACD
            df['macd'] = ta.trend.MACD(df['close']).macd()

            # Stochastic RSI
            df['stoch_rsi'] = ta.momentum.StochRSIIndicator(df['close']).stochrsi()

            # Williams %R
            df['wpr'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()

            # Ultimate Oscillator
            df['uo'] = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()

            # Money Flow Index (MFI)
            df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()

            # Force Index
            df['force_index'] = ta.volume.ForceIndexIndicator(df['close'], df['volume']).force_index()

            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.volatility.BollingerBands(
                df['close']).bollinger_hband_indicator(), ta.volatility.BollingerBands(
                df['close']).bollinger_mavg(), ta.volatility.BollingerBands(df['close']).bollinger_lband_indicator()

            # Fibonacci levels
            df['fibonacci_0.382'] = df['close'].rolling(window=20).mean() * 0.382
            df['fibonacci_0.618'] = df['close'].rolling(window=20).mean() * 0.618

            # Calculate probabilities for technical indicators
            probabilities = calculate_technical_probabilities(df)

            # Print individual probabilities with "bullish" or "bearish" in parentheses
            print("\nResolution:", resolution)
            for indicator, probability in probabilities.items():
                sentiment = 'Bullish' if probability > 0.5 else 'Bearish'
                print(f"{indicator.capitalize()} Probability: {probability:.3f} ({sentiment})")

            # Fetch quarterly financial data and calculate probabilities
            financials = fetch_quarterly_financial_data(symbol)
            financial_metrics_probabilities = calculate_financial_metrics_probabilities(financials)

            # Print individual financial metrics probabilities with "bullish" or "bearish" in parentheses
            for metric, probability in financial_metrics_probabilities.items():
                sentiment = 'Bullish' if probability > 0.5 else 'Bearish'
                print(f"{metric} Probability: {probability:.3f} ({sentiment})")

            # Combine technical and financial probabilities
            combined_probabilities = {**probabilities, **financial_metrics_probabilities}

            # Calculate bullish and bearish percentages
            bullish_percentage += sum(
                probability for indicator, probability in combined_probabilities.items() if probability > 0.5)
            bearish_percentage += sum(
                probability for indicator, probability in combined_probabilities.items() if probability <= 0.5)

            # Store the data for plotting
            historical_data_list.append(df)
            indicators_and_probabilities_list.append((resolution, combined_probabilities))

            # Calculate overall sentiment
            total_sentiment = 'Bullish' if bullish_percentage > bearish_percentage else 'Bearish'
            print("\nOverall Sentiment:", total_sentiment)
            print(f"Bullish Percentage: {bullish_percentage:.2f}%")
            print(f"Bearish Percentage: {bearish_percentage:.2f}%")

            # Calculate the final decision (bullish or bearish) and probability score (in percentage)
            final_decision = 'Bullish' if total_sentiment == 'Bullish' else 'Bearish'
            Ai_score = bullish_percentage / (bullish_percentage + bearish_percentage) * 100

            # Print the final decision and formatted probability score
            print("\nFinal Decision:", final_decision)
            print(f"AI Probability Score: {Ai_score:.2f}%")

            # Print the predicted stock price
            print(f"Resolution: {resolution}")
            print(f"Next {resolution} Future Close Price: {df['predicted_future_close'].tail(1).values[0]:.4f}")

            # Create a list to store the DataFrames for each symbol
            symbol_dfs = []

            symbols_of_choice = ['ARM', 'NVDA', 'TSGS']

            for symbol in symbols_of_choice:
                # Fetch and process symbol data as before
                df = fetch_and_process_symbol_data(symbol, resolutions, finnhub_api_key, prediction_period,
                                                   num_days_to_predict, feature_columns)

                # Define the target column
                target_column = 'next_1day_future_close'  # Assuming this is the target column you want to predict

                # Split the dataset and obtain the scaler
                X_train_scaled, X_valid_scaled, y_train, y_valid, scaler = load_and_split_dataset(df, feature_columns,
                                                                                                  target_column)

                # Append the DataFrame to the list
                symbol_dfs.append(df)

            # Concatenate the DataFrames with an additional 'symbol' column
            for i, df in enumerate(symbol_dfs):
                df['symbol'] = symbols_of_choice[i]

            # Concatenate all DataFrames into a single DataFrame
            combined_df = pd.concat(symbol_dfs, ignore_index=True)

            # Define the file path for the combined data
            combined_file_path = "/Users/teera/Downloads/future_data_combined.csv"

            # Save the combined DataFrame to a CSV file
            combined_df.to_csv(combined_file_path, index=False)

            # Print the file path
            print(f"Combined data saved to: {combined_file_path}")

            # Print the predicted stock price
            print(f"Resolution: {resolution}")
            print(f"Next {resolution} Future Close Price: {df['predicted_future_close'].tail(1).values[0]:.4f}")

            # Define the resolution for which you want to make predictions
            resolution_to_predict = 'D'  # Replace with your desired resolution

        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")

            # Find today's date
    today = pd.to_datetime('today').replace(hour=0, minute=0, second=0, microsecond=0)

    # Plotting (outside of the resolution loop)
    for df, (resolution, combined_probabilities) in zip(historical_data_list, indicators_and_probabilities_list):
        plt.figure(figsize=(12, 6))
        plt.title(f"{symbol} ({resolution}) - Technical Analysis")

        # Adjust the right margin to make space for the legend
        plt.subplots_adjust(right=0.75)  # Adjust the right margin to make space for the legend

        # Filter the historical data to include only the last 7 years
        seven_years_ago = pd.to_datetime('today') - pd.DateOffset(years=7)
        df = df[df['timestamp'] >= seven_years_ago]

        # Plot the historical close prices
        plt.plot(df['timestamp'], df['close'], label='Close Price', color='blue')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)

        # Find today's date
        today = pd.to_datetime('today').replace(hour=0, minute=0, second=0, microsecond=0)

        # Filter the future data to include only dates starting from today
        future_data = df[df['timestamp'] >= today]

        # Assuming 'next_1day_future_close' column is already present in the DataFrame 'future_data'
        # Convert the 'timestamp' column to datetime data type
        future_data['timestamp'] = pd.to_datetime(future_data['timestamp'])

        # Add one day to the 'timestamp' column
        future_data['timestamp'] = future_data['timestamp'] + pd.DateOffset(days=1)

        # Define the file path
        file_path = f"/Users/teera/Downloads/{symbol}27.csv"

        # Save the DataFrame to a CSV file
        future_data.to_csv(file_path, index=False)

        # Plot the data with the updated 'timestamp' column
        plt.plot(future_data['timestamp'], future_data['next_1day_future_close'], label='Future Close Price',
                 color='green', linestyle='--')

        # Print predicted stock price
        predicted_price = df['next_1day_future_close'].dropna().iloc[-1]
        print(f"Next 1 Day Future Close Price: {predicted_price:.4f}")

        # Adjust the legend settings to make them smaller and move them outside
        plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1.02, 1.01))  # Move legend outside

        plt.show()


# Function to fetch symbols from the database
def fetch_symbols_from_database(cursor, query):
    try:
        cursor.execute(query)
        print("SQL Query executed successfully")

        # Fetch all the ticker values
        ticker_list = cursor.fetchall()

        # Extract tickers into a list
        tickers = [ticker[0] for ticker in ticker_list]

        return tickers

    except Exception as e:
        print(f"Error executing SQL query: {str(e)}")
        return []


@retry.retry(exceptions=Exception, tries=5, delay=10)  # Retry 5 times with a delay of 10 seconds between retries
def fetch_and_process_symbol_data_with_retry(symbol, resolutions, finnhub_api_key, prediction_period,
                                             num_days_to_predict, feature_columns):
    try:
        # Add your code here to fetch and process symbol data
        fetch_and_process_symbol_data(symbol, resolutions, finnhub_api_key, prediction_period, num_days_to_predict,
                                      feature_columns)
        return df
    except Exception as e:
        print(f"Failed to fetch data for {symbol}. Error: {str(e)}")
        raise


def main():
    try:
            symbol = ['AAPL']
            resolutions = ['1', '5', '15', '30', '60', '120', '240', 'D']
            prediction_period = 1
            num_days_to_predict = 1
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume', 'rsi', 'stochastic_oscillator',
                'macd', 'cci', 'adx', 'atr', 'wpr'
            ]

                try:
                    # Fetch and process symbol data as before
                    df_symbol = fetch_and_process_symbol_data_with_retry(symbol, resolutions, finnhub_api_key,
                                                                         prediction_period, num_days_to_predict,
                                                                         feature_columns)

                    if df_symbol is not None:
                        print(f"Processing data for symbol: {symbol}")

                        # Print the first few rows of the data frame
                        print(df_symbol.head())

                        # Split the dataset and obtain the scaler
                        X_train_scaled, X_valid_scaled, y_train, y_valid, scaler = load_and_split_dataset(df_symbol,
                                                                                                          feature_columns,
                                                                                                          'next_1day_future_close')

                        # Continue with the rest of your code...
                        model = train_lstm_model(X_train_scaled, y_train, epochs=50, batch_size=32)

                        # Make predictions using the trained LSTM model
                        predictions = make_lstm_predictions(model, X_valid_scaled)

                        if len(predictions) > 0:
                            # Update the DataFrame with the predicted prices
                            df_symbol = update_dataframe_with_lstm_prediction(df_symbol, predictions)

                            # Save the trained LSTM model to a file with the symbol name (if needed)
                            model_filename = os.path.join(output_dir, f'{symbol}_lstm_model.h5')
                            model.save(model_filename)
                            print(f"Saved LSTM model for {symbol} to {model_filename}")

                            # ... (continue with the rest of your code)

                except Exception as e:
                    print(f"Error processing symbol {symbol}: {e}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
