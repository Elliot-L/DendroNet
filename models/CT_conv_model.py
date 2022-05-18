import torch.nn.functional as func
import torch.nn as nn
import torch

# Model inspired by architecture found in
# https://dspace.mit.edu/bitstream/handle/1721.1/111019/Convolutional%20neural%20network%20architectures%20for%20predicting%20DNA%E2%80%93protein%20binding.pdf?sequence=1&isAllowed=y

class CTspecificConvNet(nn.Module):
    def __init__(self, cell_type, device, seq_length, kernel_size):
        super().__init__()

        """
        convolution part
        """
        self.device = device
        self.cell_type = cell_type
        self.seq_length = seq_length  # this is L that we see below
        self.kernel_size = kernel_size  # this is k that we see below

        # We take in an input of length L, with 4 channels (input shape is (n,4,L), where n is batch size)
        self.convLayer = nn.Conv1d(4, 32, self.kernel_size)
        # we get an output of length L - k + 1, with 32 channels (32 convolutions) (output shape is (n,32,L - k + 1))

        # we perform max polling for each convolution
        self.globalMaxPool = nn.MaxPool1d(self.seq_length - self.kernel_size + 1)
        # thus, output shape is (n, 32)

        # fully connected part
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x is of shape (n, 4, 501) where n is the batch size
        x = func.relu(self.convLayer(x))  # size is now (n, 32, 478)
        x = self.globalMaxPool(x)  # size is now (n, 32)
        x = torch.squeeze(x)
        # print("result of convolution: ")
        # print(x)
        x = func.relu(self.fc1(x))  # same size
        x = func.sigmoid(self.fc2(x))  # size is now n, representing final output for whole batch
        return torch.squeeze(x)

class MultiCTConvNet(nn.Module):
    def __init__(self, device, num_cell_types, seq_length, kernel_size):
        super().__init__()
        """
        convolution part
        """
        self.device = device
        self.num_cell_types = num_cell_types
        self.seq_length = seq_length  # this is L that we see below
        self.kernel_size = kernel_size  # this is k that we see below

        # We take in an input of length L, with 4 channels (input shape is (n,4, L), where n is batch size)
        self.convLayer = nn.Conv1d(4, 32, self.kernel_size)
        # we get an output of length L - k + 1, with 32 channels (32 convolutions) (output shape is (n,32,L - k + 1))

        # we perform max pooling for each convolution
        self.globalMaxPool = nn.MaxPool1d(self.seq_length - self.kernel_size + 1)
        # thus, output shape is (n, 32)

        # fully connected part
        self.fc1 = nn.Linear(32 + self.num_cell_types, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, cell_types):
        # x is of shape (n, 4, 501) where n is the batch size
        # cell_types is of shape (n, num_cell_types)
        print(x.size())
        x = func.relu(self.convLayer(x))  # size is not (n, 32, 478)
        print(x.size())
        x = self.globalMaxPool(x)  # size is now (n, 1, 32), this is why we need to squeeze
        print(x.size())
        x = torch.squeeze(x)
        print(x.size())
        x = torch.cat((x, cell_types), 1)  # x is now (n, 32 + num_cell_types)
        print(x.size())
        x = func.relu(self.fc1(x))  # same size
        print(x.size())
        x = func.sigmoid(self.fc2(x))  # size is now n, representing final output for whole batch
        return torch.squeeze(x)