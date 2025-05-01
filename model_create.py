import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2,hidden_size3,dropout_rate, out_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc_out = nn.Linear(hidden_size3, out_size)
        self.batch1 = nn.BatchNorm1d(hidden_size1)
        self.batch2 = nn.BatchNorm1d(hidden_size2)
        self.batch3 = nn.BatchNorm1d(hidden_size3)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.fc1(x)
        out = self.batch1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.batch2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.batch3(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc_out(out)
        return out

if __name__ == "__main__":
    print("just model structure")