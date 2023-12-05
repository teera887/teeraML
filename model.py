import torch.nn as nn


class HelloLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, fc_unites=48, seed=2023):
        super(HelloLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=output_dim,
            num_layers=3,
            dropout=0.3,
            bidirectional=False,
            batch_first=True
        )

        self.fc1 = nn.Linear(output_dim, fc_unites)
        self.fc2 = nn.Linear(fc_unites, output_dim)

    def forward(self, input_data):
        output_lstm, (hs, cs) = self.lstm(input_data)
        output_fc1 = self.fc1(output_lstm)
        return self.fc2(output_fc1)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(-3e-2, 3e-2)
        self.fc2.weight.data.uniform_(-3e-2, 3e-2)
