import ta

from utils.data_cracker import fetch_quarterly_financial_data


def calculate_financial_metrics_probabilities(financials):
    probabilities = {}
    if financials is not None:
        # Calculate the percentage & the probability of increment for each financial metric
        metrics = [
            'Basic EPS',
            'Normalized EBITDA',
            'EBIT',
            'Net Income Continuous Operations',
            'Gross Profit',
            'Diluted EPS',
            'Reconciled Cost Of Revenue',
            'Total Revenue',
            'Operating Revenue',
            'Total Operating Income as Reported',
            'Total Expenses',
            'Basic Average Shares',
            'Cost Of Revenue',
            'Operating Expense',
            'Interest Expense Non Operating',
            'Selling General And Administration',
            'Research And Development',
            'Interest Income Non Operating',
            'Pretax Income',
            'Tax Provision',
            'Tax Rate For Calcs',
            'Diluted NI Availto Com Stockholders',
            'Net Non Operating Interest Income Expense',
            'Other Income Expense',
        ]

    for metric in metrics:
        try:
            metric_table = financials.loc[metric]
            percentage_increment = ((metric_table - metric_table.shift(1)) / metric_table.shift(1)) * 100
            probability = len(percentage_increment[percentage_increment > 0]) / len(percentage_increment)
            probabilities[metric] = probability

            # Calculate the sentiment label based on probability
            if probability >= 0.5:
                sentiment_label = "Bullish"
            else:
                sentiment_label = "Bearish"

            probabilities[metric] = probability

            print(f'Percentage Increment of {metric}: {percentage_increment.mean():.2f}%')
            print(f'Probability of {metric} Increment: {probability:.3f} ({sentiment_label})')

        except Exception as e:
            print(f"Error calculating {metric} probability: {e}")
            probabilities[metric] = 0  # Default to neutral if calculation fails

    return probabilities


def calculate_technical_probabilities(predicted_price, financial_metrics_probabilities):
    probabilities = {}
    bullish_percentage = 0
    bearish_percentage = 0

    # RSI
    predicted_price['rsi'] = ta.momentum.RSIIndicator(predicted_price['close']).rsi()

    # Stochastic %K
    predicted_price['stoch_k'] = ta.momentum.StochasticOscillator(predicted_price['high'], predicted_price['low'], predicted_price['close']).stoch()

    # CCI
    predicted_price['cci'] = ta.trend.cci(predicted_price['high'], predicted_price['low'], predicted_price['close'])

    # ADX
    predicted_price['adx'] = ta.trend.ADXIndicator(predicted_price['high'], predicted_price['low'], predicted_price['close']).adx()

    # Awesome Oscillator
    predicted_price['ao'] = ta.momentum.AwesomeOscillatorIndicator(predicted_price['high'], predicted_price['low']).awesome_oscillator()

    # Momentum (10)
    predicted_price['momentum_1'] = ta.momentum.roc(predicted_price['close'], 1)

    # MACD
    predicted_price['macd'] = ta.trend.MACD(predicted_price['close']).macd()

    # Stochastic RSI
    predicted_price['stoch_rsi'] = ta.momentum.StochRSIIndicator(predicted_price['close']).stochrsi()

    # Williams %R
    predicted_price['wpr'] = ta.momentum.WilliamsRIndicator(predicted_price['high'], predicted_price['low'], predicted_price['close']).williams_r()

    # Ultimate Oscillator
    predicted_price['uo'] = ta.momentum.UltimateOscillator(predicted_price['high'], predicted_price['low'], predicted_price['close']).ultimate_oscillator()

    # Money Flow Index (MFI)
    predicted_price['mfi'] = ta.volume.MFIIndicator(predicted_price['high'], predicted_price['low'], predicted_price['close'], predicted_price['volume']).money_flow_index()

    # Force Index
    predicted_price['force_index'] = ta.volume.ForceIndexIndicator(predicted_price['close'], predicted_price['volume']).force_index()

    # Bollinger Bands
    predicted_price['bb_upper'], predicted_price['bb_middle'], predicted_price['bb_lower'] = ta.volatility.BollingerBands(
        predicted_price['close']).bollinger_hband_indicator(), ta.volatility.BollingerBands(
        predicted_price['close']).bollinger_mavg(), ta.volatility.BollingerBands(predicted_price['close']).bollinger_lband_indicator()

    # Fibonacci levels
    predicted_price['fibonacci_0.382'] = predicted_price['close'].rolling(window=20).mean() * 0.382
    predicted_price['fibonacci_0.618'] = predicted_price['close'].rolling(window=20).mean() * 0.618

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
            indicator_values = predicted_price[indicator].dropna()

            if len(indicator_values) > 0:
                indicator_probability = len(indicator_values[indicator_values > threshold]) / len(indicator_values)
                probabilities[indicator] = indicator_probability
            else:
                probabilities[indicator] = 0.5  # Default to neutral if no data

        except Exception as e:
            print(f"Error calculating {indicator} probability: {e}")
            probabilities[indicator] = 0.5  # Default to neutral

            # Combine technical and financial probabilities
            combined_probabilities = {**probabilities, **financial_metrics_probabilities}

            # Calculate bullish and bearish percentages
            bullish_percentage += sum(
                probability for indicator, probability in combined_probabilities.items() if probability > 0.5)
            bearish_percentage += sum(
                probability for indicator, probability in combined_probabilities.items() if probability <= 0.5)

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


if __name__ == '__main__':
    finance_data = fetch_quarterly_financial_data("NVDA")
    finance_matrx_prob = calculate_financial_metrics_probabilities(finance_data)
