from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter

from module import LSTM
from sklearn.preprocessing import MinMaxScaler

data_ = pd.read_csv("dataset/NVDA.csv")
scaler = MinMaxScaler(feature_range=(-1, 1))
data_['c'] = scaler.fit_transform(data_['c'].values.reshape(-1,1))
NVDA_data = data_['c']


# function to create train, test data given stock data and sequence length
def load_data(stock, lookback):
    data_raw = stock.values  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data)
    data = data.reshape([*data.shape, 1])
    test_set_size = int(np.round(0.05 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]


look_back = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x_train, y_train, x_test, y_test = load_data(NVDA_data, look_back)
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)

# make training and test sets in torch
x_train = torch.from_numpy(x_train).type(torch.Tensor).to(device)
x_test = torch.from_numpy(x_test).type(torch.Tensor).to(device)
y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)
y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)

n_steps = look_back - 1
batch_size = 2048
num_epochs = 300

train = torch.utils.data.TensorDataset(x_train, y_train)
test = torch.utils.data.TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=batch_size,
                                          shuffle=False)

### initialize model

input_dim = 1
hidden_dim = 64
num_layers = 2
output_dim = 1

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model.to(device)
if torch.cuda.device_count() > 1:
    print("Using multiple GPUs")
    model = DataParallel(model)

loss_fn = torch.nn.MSELoss(size_average=True)

optimiser = torch.optim.RMSprop(model.parameters(), lr=0.01)

writer = SummaryWriter()

print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

# %%
# Train model

for epoch in range(num_epochs):
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    train_loss = []

    for batch_idx, (data, targets) in progress_bar:
        # Forward pass
        y_train_pred = model(data)

        loss = loss_fn(y_train_pred, targets)

        # store loss data
        train_loss.append(loss.item())

        progress_bar.set_description(
            f'Epoch {epoch + 1}/{num_epochs} - Batch {batch_idx + 1}/{len(train_loader)} MSE: {loss.item()}')

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()

    writer.add_scalar("Loss/Train", sum(train_loss) / len(train_loss), epoch)

    # test the model
    y_test_pred = model(x_test)
    loss_test = loss_fn(y_test_pred, y_test)
    writer.add_scalar('Loss/Test', loss_test.item(), epoch)

    # save model every 1000 epochs
    if (epoch + 1) % 100 == 0:
        torch.save(model, f"checkpoint{epoch}.pth")

    writer.flush()
