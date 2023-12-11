import ta

from utils.data_cracker import *


def calculate_financial_metrics_probabilities(financials):
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
        "now let's see how it goes" "ok" "seems like its gonna take forever hahah nope see loss is decreasing significantly"
    ]
    probabilities = {}

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

    return probabilities


def process_historical_data(df):
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

    return df


if __name__ == '__main__':
    symbols = ['NVDA']
    resolutions = ['1', '5', '15', '30', '60', '120', '240', 'D']

    historical_data = fetch_historical_data(symbols[0], resolutions[0], api_key=config.finnhub_api_key)
    financial_data = fetch_quarterly_financial_data(symbols[0])
    processed_historical_data = process_historical_data(historical_data)



