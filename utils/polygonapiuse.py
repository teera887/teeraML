import datetime

import pandas as pd
import requests

api_key = "v52yhzptchynW6h3xFf2xu8hnOaXPfdV"

symbol = "NVDA"
start_date = '2023-12-01'
end_date = '2023-12-3'
timestamp_start = datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp()
timestamp_end = datetime.datetime.strptime(end_date, "%Y-%m-%d").timestamp()
idx_timestamp = timestamp_start
data = pd.DataFrame()
base_url = "https://api.polygon.io/v2/aggs/ticker"
multiplier = 1
unit = 'minute'
result = pd.DataFrame()

while True:
    if idx_timestamp > timestamp_end:
        break
    current_date = datetime.datetime.fromtimestamp(idx_timestamp).strftime("%Y-%m-%d")
    tmr_timestamp = (datetime.datetime.fromtimestamp(idx_timestamp) + datetime.timedelta(days=1)).timestamp()
    tmr_date = datetime.datetime.fromtimestamp(tmr_timestamp).strftime("%Y-%m-%d")
    request_url = "/{}/range/{}/{}/{}/{}?apiKey={}".format(symbol, multiplier, unit, current_date, tmr_date,
                                                           api_key)
    response = requests.get(base_url + request_url)
    if response.status_code == 200:
        try:
            data = response.json()['results']
            tmp_df = pd.DataFrame(data)
            tmp_df.t = pd.to_datetime(tmp_df.t, unit='ms', utc=True)
            result = result._append(tmp_df)
        except:
            print(f"No data detected from {current_date}")
    else:
        print(f"fetching error...passing {current_date}'s data...")
    idx_timestamp = tmr_timestamp

result = result.drop_duplicates()
result.set_index('t', inplace=True)
result.to_csv("dataset/NVDA_recent.csv")
