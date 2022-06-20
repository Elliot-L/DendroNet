import torch.nn.functional as func
import torch.nn as nn
import torch

# Model inspired by architecture found in
# https://dspace.mit.edu/bitstream/handle/1721.1/111019/Convolutional%20neural%20network%20architectures%20for%20predicting%20DNA%E2%80%93protein%20binding.pdf?sequence=1&isAllowed=y

class SimpleCTspecificConvNet(nn.Module):
    def __init__(self, cell_type, device, seq_length, kernel_size, num_of_kernels, polling_window, initial_channels=4):
        super().__init__()
        """
        convolution part (1 layers)
        """
        self.device = device
        self.cell_type = cell_type
        self.seq_length = seq_length  # this is L that we see below
        self.kernel_size = kernel_size  # this is k that we see below
        self.num_of_kernels = num_of_kernels  # this is m
        if polling_window != 0:
            self.polling_window = polling_window  # this is p
        else:
            self.polling_window = self.seq_length - self.kernel_size + 1  # or this is p
        self.initial_channels = initial_channels

        # We take in an input of length L, with 4 channels (input shape is (n,4,L), where n is batch size)
        self.convLayer1 = nn.Conv1d(self.initial_channels, self.num_of_kernels, self.kernel_size, device=device)
        # we get an output of length L - k1 + 1, with m1 channels(number of kernels), output shape is (n,m,L - k + 1)

        # we perform max polling for each convolution
        self.MaxPoolLayer1 = nn.MaxPool1d(self.polling_window)
        # thus, output shape is (n, m, (L - k + 1)/p)

        # Flattening output of convolution to make it 2D
        self.flatten = nn.Flatten()
        # new shape is (n, m*((L - k + 1)/p))
        """
        fully connected part
        """
        self.fc1 = nn.Linear(self.num_of_kernels*(int((self.seq_length - self.kernel_size + 1)/self.polling_window)),
                             32, device=device)
        self.fc2 = nn.Linear(32, 1, device=device)

    def forward(self, x):
        p = False
        # x is of shape (n, 4, L) where n is the batch size
        if p:
            print(x.size())
        x = func.relu(self.convLayer1(x))
        if p:
            print(x.size())
        x = self.MaxPoolLayer1(x)
        if p:
            print(x.size())
        x = self.flatten(x)
        if p:
            print(x.size())
        x = func.relu(self.fc1(x))
        if p:
            print(x.size())
        x = torch.sigmoid(self.fc2(x))  # size is now n, representing final output for whole batch
        if p:
            print(x.size())
        return torch.squeeze(x)

class CTspecificConvNet(nn.Module):
    def __init__(self, cell_type, device, seq_length, kernel_size, num_of_kernels, polling_window, initial_channels=4):
        super().__init__()
        """
        convolution part (3 layers)
        """
        self.device = device
        self.cell_type = cell_type
        self.seq_length = seq_length  # this is L that we see below
        self.kernel_size = kernel_size  # this is (k1,k2,k3) that we see below
        self.num_of_kernels = num_of_kernels  # this is (m1,m2,m3)

        self.polling_window = (polling_window[0], polling_window[1],
                               (int(((int((self.seq_length - self.kernel_size[0] + 1)/polling_window[0]))
                                - self.kernel_size[1] + 1)/polling_window[1])) - self.kernel_size[2] + 1)
        self.initial_channels = initial_channels
        # this is (p1,p2,p3)
        # (p3 = ((((L - k1 + 1)/p1) - k2 + 1)/p2) - k3 + 1
        # which is the length of the output for the second convolution)

        # We take in an input of length L, with 4 channels (input shape is (n,4,L), where n is batch size)
        self.convLayer1 = nn.Conv1d(self.initial_channels, self.num_of_kernels[0],
                                    self.kernel_size[0], device=device)
        # we get an output of length L - k1 + 1, with m1 channels(number of kernels), output shape is (n,m1,L - k1 + 1)

        # we perform local max polling for each convolution
        self.MaxPoolLayer1 = nn.MaxPool1d(self.polling_window[0])
        # thus, output shape is (n, m1, (L - k1 + 1)/p1)

        # we perform a second convolutional operation
        self.convLayer2 = nn.Conv1d(self.num_of_kernels[0], self.num_of_kernels[1],
                                    self.kernel_size[1], device=device)
        # output here has shape (n, m2, ((L - k1 + 1)/p1) - k2 + 1)

        # we perform a second local max polling on the second convolution result
        self.MaxPoolLayer2 = nn.MaxPool1d(self.polling_window[1])
        # output here has shape (n, m2, (((L - k1 + 1)/p1) - k2 + 1)/p2)

        # we perform a final convolutional operation
        self.convLayer3 = nn.Conv1d(self.num_of_kernels[1], self.num_of_kernels[2],
                                    self.kernel_size[2], device=device)
        # output here has shape (n, m3, ((((L - k1 + 1)/p1) - k2 + 1)/p2) - k3 + 1)

        # we perform a global max pooling to
        self.MaxPoolLayer3 = nn.MaxPool1d(self.polling_window[2])
        # output has shape (n, m3, 1)

        # Flattening output of convolution to make it 2D
        self.flatten = nn.Flatten()
        # new shape is (n, m3)

        """
        fully connected part
        """

        self.fc1 = nn.Linear(self.num_of_kernels[2], 32, device=device)
        self.fc2 = nn.Linear(32, 1, device=device)

    def forward(self, x):
        p = False
        # x is of shape (n, 4, L) where n is the batch size
        if p:
            print(x.size())
        x = func.relu(self.convLayer1(x))
        if p:
            print(x.size())
        x = self.MaxPoolLayer1(x)
        if p:
            print(x.size())
        x = func.relu(self.convLayer2(x))
        if p:
            print(x.size())
        x = self.MaxPoolLayer2(x)
        if p:
            print(x.size())
        x = func.relu(self.convLayer3(x))
        if p:
            print(x.size())
        x = self.MaxPoolLayer3(x)
        if p:
            print(x.size())
        x = self.flatten(x)
        if p:
            print(x.size())
        x = func.relu(self.fc1(x))
        if p:
            print(x.size())
        x = torch.sigmoid(self.fc2(x))  # size is now n, representing final output for whole batch
        if p:
            print(x.size())
        return torch.squeeze(x)

class MultiCTConvNet(nn.Module):
    def __init__(self, device, num_cell_types, seq_length, kernel_size, number_of_kernels, polling_window):
        super().__init__()
        # If polling_window = 0, a global max polling operation will be performed
        """
        convolution part
        """
        self.device = device
        self.num_cell_types = num_cell_types
        self.seq_length = seq_length  # this is L that we see below
        self.kernel_size = kernel_size  # this is k that we see below
        self.num_of_kernels = number_of_kernels  # this is m
        if polling_window == 0:
            self.polling_window = self.seq_length - self.kernel_size + 1  # this is p
        else:
            self.polling_window = polling_window  # this is p


        # We take in an input of length L, with 4 channels (input shape is (n,4, L), where n is batch size)
        self.convLayer = nn.Conv1d(4, self.num_of_kernels, self.kernel_size, device=device)
        # we get an output of length L - k + 1, with m channels (number of kernels) (output shape is (n,m , L - k + 1))

        # we perform max pooling for each convolution without overlap (stride is size of polling window)
        self.globalMaxPool = nn.MaxPool1d(self.polling_window)
        # thus, output shape is (n, m, (L - k + 1)/p)

        # Flattening output of convolution to make it 2D
        self.flatten = nn.Flatten()
        # new shape is (n, m((L - k + 1)/p))

        # fully connected part
        # We add the size of the cell type vector (one hot encoding, to the inputted feature vector)
        self.fc1 = nn.Linear(self.num_of_kernels*(int((self.seq_length - self.kernel_size + 1)/self.polling_window))
                             + self.num_cell_types, 32, device=device)
        self.fc2 = nn.Linear(32, 1, device=device)

    def forward(self, x, cell_types):
        p = False
        # x is of shape (n, 4, 501) where n is the batch size
        # cell_types is of shape (n, num_cell_types)
        if p:
            print(x.size())
        x = func.relu(self.convLayer(x))  # size is not (n, 32, 478)
        if p:
            print(x.size())
        x = self.globalMaxPool(x)  # size is now (n, 1, 32), this is why we need to squeeze
        if p:
            print(x.size())
        x = self.flatten(x)
        if p:
            print(x.size())
        x = torch.cat((x, cell_types), 1)  # x is now (n, 32 + num_cell_types)
        if p:
            print(x.size())
        x = func.relu(self.fc1(x))  # same size
        if p:
            print(x.size())
        x = torch.sigmoid(self.fc2(x))  # size is now n, representing final output for whole batch
        if p:
            print(x.size())
        return torch.squeeze(x)

class SeqConvModule(nn.Module):
    def __init__(self, device, seq_length, kernel_sizes, num_of_kernels,
                 polling_windows, input_channels):
        """
        by default, the last polling layer is a global polling
        """
        super().__init__()
        self.device = device
        self.seq_length = seq_length
        self.kernel_sizes = kernel_sizes
        self.num_of_kernels = num_of_kernels
        self.num_of_layers = len(kernel_sizes)
        self.polling_windows = list(polling_windows[0: self.num_of_layers])
        self.polling_windows.append((int(((int((self.seq_length - self.kernel_sizes[0] + 1)/polling_windows[0]))
                                     - self.kernel_sizes[1] + 1)/polling_windows[1])) - self.kernel_sizes[2] + 1)
        self.input_channels = input_channels

        self.conv_layer_list = nn.ModuleList()
        self.polling_layer_list = nn.ModuleList()

        for layer in range(self.num_of_layers):
            if layer == 0:
                conv_layer = nn.Conv1d(self.input_channels, self.num_of_kernels[layer],
                                       self.kernel_sizes[layer], device=device)
            else:
                conv_layer = nn.Conv1d(self.num_of_kernels[layer-1], self.num_of_kernels[layer],
                                       self.kernel_sizes[layer], device=device)
            poll_layer = nn.MaxPool1d(self.polling_windows[layer])

            self.conv_layer_list.append(conv_layer)
            self.polling_layer_list.append(poll_layer)

    def forward(self, seq):
        #print('Convolutional component')
        p = False
        for i in range(self.num_of_layers):
            conv_layer = self.conv_layer_list[i]
            poll_layer = self.polling_layer_list[i]
            seq = func.relu(conv_layer(seq))
            if p:
                print(seq.size())
            seq = poll_layer(seq)
            if p:
                print(seq.size())
        return torch.squeeze(seq)


class DendronetModule(nn.Module):
    def __init__(self, device, root_weights, path_mat, delta_mat, p=1, init_deltas=False, init_root=True):
        """
        #param p: type of norm to take for dendronet loss
        """
        super().__init__()
        self.device = device
        self.path_mat = torch.tensor(path_mat, device=device, dtype=torch.double)
        self.p = p
        self.root_weights = nn.Parameter(torch.tensor(root_weights, device=device, dtype=torch.double,
                                                      requires_grad=True))
        if init_root:
            torch.nn.init.normal_(self.root_weights, mean=0.0, std=0.01)

        self.delta_mat = nn.Parameter(torch.tensor(delta_mat, device=device, dtype=torch.double,
                                                   requires_grad=True))
        if init_deltas:
            torch.nn.init.normal_(self.delta_mat, mean=0.0, std=0.01)

    def delta_loss(self, idx=None):
        if idx is not None:
            edges = torch.max(self.path_mat[:, idx], dim=1)
            mat_slice = self.delta_mat.T[torch.nonzero(edges.values == 1.0).reshape(-1)]
            return torch.norm(mat_slice, p=self.p)

        return torch.norm(self.delta_mat, p=self.p)

    def root_loss(self):
        root_loss = 0.0
        for w in self.root_weights:
            root_loss += abs(float(w))
        return root_loss

    def forward(self, node_idx):
        #print('Dendronet component:')
        p = False
        self.path_mat = self.path_mat.to(self.device)
        embeddings = torch.add(self.root_weights, torch.matmul(self.delta_mat, self.path_mat[:, node_idx]).T)
        if p:
            print(embeddings.size())
        return embeddings

    def get_embedding(self, node_idx):
        embeddings = torch.add(self.root_weights, torch.matmul(self.delta_mat, self.path_mat[:, node_idx]).T)
        return embeddings


class FCModule(nn.Module):
    def __init__(self, device, layer_sizes=(32, 32, 1)):
        super().__init__()
        self.layer_sizes = layer_sizes

        self.flatten = nn.Flatten()
        self.linear_layers = nn.ModuleList()

        for layer in range(len(layer_sizes) - 1):
            lin_layer = self.fc1 = nn.Linear(layer_sizes[layer], layer_sizes[layer + 1], device=device)
            self.linear_layers.append(lin_layer)

    def forward(self, x):
        #print('Fully connected component:')
        print(x.device)
        # x = x.type(torch.FloatTensor)
        p = False
        print(x.device)
        x = self.flatten(x)
        if p:
            print(x.size())
        print(x.device)

        for layer in range(len(self.linear_layers) - 1):
            x = func.relu(self.linear_layers[layer](x))
            if p:
                print(x.size())

        x = torch.sigmoid(self.linear_layers[-1](x))
        if p:
            print(x.size())

        return torch.squeeze(x)





