import requests
import pandas as pd
from datetime import timedelta, date

api_key = "v52yhzptchynW6h3xFf2xu8hnOaXPfdV"


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def get_historical_data(apiKey="", symbol="AAPL", multiplier=1, timespan='minute', start_date=date(2020, 1, 1),
                        end_date=date(2023, 11, 29)):
    base_url = 'https://api.polygon.io/v2/'
    df_tmp = pd.DataFrame()

    for single_date in daterange(start_date, end_date):
        from_date = single_date.strftime("%Y-%m-%d")
        print(from_date)
        to_date = (single_date + timedelta(days=1)).strftime("%Y-%m-%d")

        endpoint = "aggs/ticker/{}/range/{}/{}/{}/{}?apiKey={}".format(symbol, multiplier
                                                                       , timespan, from_date, to_date, apiKey)

        response = requests.get(base_url + endpoint)

        if response.status_code == 200:
            data = response.json()['results']
            single_day_df = pd.DataFrame(data)
            single_day_df['t'] = pd.to_datetime(single_day_df['t'], unit='ms')
            single_day_df.set_index('t', inplace=True)
            df_tmp = df_tmp._append(single_day_df)

        else:
            print("Error from PolygonAPI")

    return df_tmp


if __name__ == '__main__':
    df = get_historical_data(apiKey=api_key)
    print(df.head())
