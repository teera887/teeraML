import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from module import LSTM

from sklearn.preprocessing import MinMaxScaler

data_ = pd.read_csv("dataset/NVDA_recent.csv")
scaler = MinMaxScaler(feature_range=(-1, 1))
data_['c'] = scaler.fit_transform(data_['c'].values.reshape(-1,1))
NVDA_data = data_['c']

input_dim = 1
hidden_dim = 64
num_layers = 2
output_dim = 1

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model.load_state_dict(torch.load('checkpoint/checkpoint299.pth').state_dict())

lookback = 40

data_raw = NVDA_data.values  # convert to numpy array
data = []

# create all possible sequences of length seq_len
for index in range(len(data_raw) - lookback):
    data.append(data_raw[index: index + lookback])

data_basic = data[:-300]

data_basic = np.array(data_basic)
data_basic = torch.tensor(data_basic.reshape([*data_basic.shape, 1]), dtype=torch.float32)

y_predict = model(data_basic)

predict = pd.DataFrame(scaler.inverse_transform(y_predict.detach().numpy()))
origin = pd.DataFrame(scaler.inverse_transform(data_raw[40:40+data_basic.shape[0]].reshape(-1, 1)))

plt.figure()
#plt.ylim(0, 600)
plt.plot(origin, label='origin')
plt.plot(predict, label='predict')
plt.show()

